package lamp.tgnn

import cats.parse.Rfc5234
import cats.parse.Parser
import cats.data.NonEmptyList
import scala.collection.immutable
import org.saddle.index.InnerJoin
import org.saddle.index.RightJoin
import org.saddle.index.OuterJoin
import org.saddle.index.LeftJoin
import lamp.tgnn.QueryDsl.SyntaxTree.ColumnRef

object QueryDsl {

  def compile(
      s: String
  )(userDefinedFuntions: List[LiftFunction] = Nil): Either[String, Result] =
    parse(s).left
      .map(_.toString)
      .flatMap(liftExpressionList(_, builtInFunctions ++ userDefinedFuntions))

  trait LiftFunction {
    def tryLift(
        name: String,
        args: NonEmptyList[LiftedArgument]
    ): Option[LiftedArgument]
  }
  object LiftFunction {
    def apply(
        fun: PartialFunction[
          (String, NonEmptyList[LiftedArgument]),
          LiftedArgument
        ]
    ) = new LiftFunction {
      def tryLift(
          name: String,
          args: NonEmptyList[LiftedArgument]
      ): Option[LiftedArgument] =
        fun.lift((name, args))
    }
  }

  val builtInFunctions = List(
    LiftFunction {
      case ("||", args)
          if args.head.isBooleanFactor && args.size == 2 && args.tail.head.isBooleanFactor =>
        LiftedBooleanFactor(
          args.head.asBooleanFactor.or(args.tail.head.asBooleanFactor)
        )
      case ("&&", args)
          if args.head.isBooleanFactor && args.size == 2 && args.tail.head.isBooleanFactor =>
        LiftedBooleanFactor(
          args.head.asBooleanFactor.and(args.tail.head.asBooleanFactor)
        )
      case ("not", args) if args.head.isBooleanFactor && args.size == 1 =>
        LiftedBooleanFactor(args.head.asBooleanFactor.negate)
      case ("identity" | "select", args) if args.head.isColumnRef =>
        LiftedBooleanFactor(args.head.asColumnRef.select)
      case ("isna" | "ismissing" | "missing", args) if args.head.isColumnRef =>
        LiftedBooleanFactor(args.head.asColumnRef.isMissing)
      case ("notna" | "isnotna" | "nonmissing", args)
          if args.head.isColumnRef =>
        LiftedBooleanFactor(args.head.asColumnRef.isNotMissing)
      case ("==", args) if args.size == 2 && args.head.isColumnRef =>
        args.tail.head match {
          case LiftedColumnRef(cr2) =>
            LiftedBooleanFactor(args.head.asColumnRef.===(cr2))
          case LiftedVariableRef(vr2) =>
            LiftedBooleanFactor(args.head.asColumnRef.===(vr2))
          case LiftedBooleanFactor(bf) =>
            LiftedBooleanFactor(args.head.asColumnRef.select.===(bf))
        }
    }
  )

  private[tgnn] def parse(
      s: String
  ): Either[Parser.Error, SyntaxTree.ExpressionList] =
    DslParser.program.parseAll(s)

  sealed trait LiftedArgument { this: LiftedArgument =>
    def asColumnRef = this.asInstanceOf[LiftedColumnRef].cr
    def isColumnRef = this.isInstanceOf[LiftedColumnRef]
    def asVariable = this.asInstanceOf[LiftedVariableRef].vr
    def isVariable = this.isInstanceOf[LiftedVariableRef]
    def asBooleanFactor =
      this.asInstanceOf[LiftedBooleanFactor].booleanFactor
    def isBooleanFactor = this.isInstanceOf[LiftedBooleanFactor]
  }
  case class LiftedColumnRef(cr: RelationalAlgebra.TableColumnRef)
      extends LiftedArgument
  case class LiftedVariableRef(vr: RelationalAlgebra.VariableRef)
      extends LiftedArgument
  case class LiftedBooleanFactor(booleanFactor: BooleanFactor)
      extends LiftedArgument

  private def liftColumnRef(cr: SyntaxTree.ColumnRef): LiftedColumnRef =
    if (cr.tableRef.isEmpty)
      LiftedColumnRef(
        RelationalAlgebra.WildcardColumnRef(
          RelationalAlgebra.StringColumnRef(cr.columnRef)
        )
      )
    else
      LiftedColumnRef(
        RelationalAlgebra.QualifiedTableColumnRef(
          RelationalAlgebra.AliasTableRef(cr.tableRef.get),
          RelationalAlgebra.StringColumnRef(cr.columnRef)
        )
      )

  private def liftArgument(
      args: SyntaxTree.Argument,
      definedFunctions: List[LiftFunction]
  ): LiftedArgument =
    args match {
      case cr: SyntaxTree.ColumnRef
          if cr.tableRef.isEmpty && cr.columnRef.toLowerCase == "true" =>
        LiftedBooleanFactor(BooleanAtomTrue)
      case cr: SyntaxTree.ColumnRef
          if cr.tableRef.isEmpty && cr.columnRef.toLowerCase == "false" =>
        LiftedBooleanFactor(BooleanAtomFalse)
      case cr: SyntaxTree.ColumnRef =>
        liftColumnRef(cr)
      case SyntaxTree.VariableRef(name) =>
        LiftedVariableRef(RelationalAlgebra.VariableRef(name))
      case SyntaxTree.FunctionWithArgs(name, args) =>
        val liftedArgs = args.map(liftArgument(_, definedFunctions))

        typeAndLiftFunction(name, liftedArgs, definedFunctions) match {
          case Right(x) => x
          case Left(s)  => throw new RuntimeException(s)
        }

    }

  private def typeAndLiftFunction(
      name: String,
      args: NonEmptyList[LiftedArgument],
      candidates: List[LiftFunction]
  ): Either[String, LiftedArgument] =
    candidates.map(_.tryLift(name, args)).find(_.isDefined).flatten match {
      case Some(value) => Right(value)
      case None =>
        Left(
          s"Implementation for '$name' with args [${args.toList.mkString(", ")}] not found. Operators are left associative without precedence rules. Variables must be the right operand."
        )
    }

  private def resolveToken(
      lexicalToken: SyntaxTree.Token,
      namedExpressions: Map[String, SyntaxTree.Expression],
      currentExpressionName: Option[String],
      definedFunctions: List[LiftFunction]
  ): Vector[StackToken] = {
    lexicalToken.name.toLowerCase match {
      case "filter" =>
        val filterExpression = liftArgument(
          lexicalToken.arguments.get.args.head.arg,
          definedFunctions
        ) match {
          case LiftedColumnRef(cr) => cr.select
          case LiftedVariableRef(_) =>
            throw new RuntimeException(
              "filter's argument can't be a variable"
            )
          case LiftedBooleanFactor(booleanFactor) => booleanFactor
        }
        Vector(StackOp1Token("filter", in => Filter(in, filterExpression)))
      case "project" =>
        val arguments = lexicalToken.arguments.get.args.map{ case SyntaxTree.ArgumentWithAlias(arg,alias0) =>
          liftArgument(arg, definedFunctions) match {
            case LiftedColumnRef(
                  cr: RelationalAlgebra.QualifiedTableColumnRef
                ) =>
              val alias = alias0
                .map(liftColumnRef)
                .collect {
                  case LiftedColumnRef(
                        x: RelationalAlgebra.QualifiedTableColumnRef
                      ) =>
                    x
                  case LiftedColumnRef(
                        x: RelationalAlgebra.WildcardColumnRef
                      ) =>
                    RelationalAlgebra
                      .QualifiedTableColumnRef(cr.table, x.column)
                }
                .getOrElse(cr)
              cr.select.as(alias)
            case LiftedVariableRef(_) =>
              throw new RuntimeException(
                "project's argument can't be a variable"
              )
            case LiftedBooleanFactor(booleanFactor: ColumnFunction) =>
              val alias = alias0
                .map(liftColumnRef)
                .collect {
                  case LiftedColumnRef(
                        x: RelationalAlgebra.QualifiedTableColumnRef
                      ) =>
                    x
                  case LiftedColumnRef(
                        x: RelationalAlgebra.WildcardColumnRef
                      ) =>
                    RelationalAlgebra
                      .QualifiedTableColumnRef(Q.table("derived"), x.column)
                }
                .getOrElse(Q.table("derived").col(booleanFactor.name))
              booleanFactor.as(alias)
            case _ =>
              throw new RuntimeException(
                s"Unexpected type for projection argument $arg"
              )
          }
        }
        Vector(
          StackOp1Token("projection", in => Projection(in, arguments.toList))
        )
      case "inner-join" | "outer-join" | "left-join" | "right-join" =>
        require(
          lexicalToken.arguments.get.args.size == 2,
          "join needs 2 arguments: join column 1, join column 2"
        )
        val arguments = lexicalToken.arguments.get.args.map(_.arg).map(arg =>
          liftArgument(arg, definedFunctions) match {
            case LiftedColumnRef(
                  cr
                ) =>
              cr
            case _ =>
              throw new RuntimeException(
                s"Unexpected type for join's argument: $arg . Two arguments expected, both should be column refs."
              )
          }
        )
        val joinType = lexicalToken.name.toLowerCase match {
          case "inner-join" => InnerJoin
          case "outer-join" => OuterJoin
          case "left-join"  => LeftJoin
          case "right-join" => RightJoin
        }
        Vector(
          StackOp2Token(
            "inner-join",
            (in1, in2) =>
              EquiJoin(in1, in2, arguments.head, arguments.tail.head, joinType)
          )
        )
      case "product" =>
        Vector(StackOp2Token("product", (in1, in2) => Product(in1, in2)))
      case "table" | "scan" | "query" =>
        val variable = lexicalToken.arguments.get.args.head.arg match {
          case SyntaxTree.VariableRef(name) => name
          case _ =>
            throw new RuntimeException(
              "the first argument of table/scan/query must be a variable"
            )
        }
        val tableReference = lexicalToken.arguments.get.args.head.alias
          .collect { case ColumnRef(None, columnRef) =>
            columnRef
          }
          .getOrElse(variable)
        val schema = {
          val args = lexicalToken.arguments.get.args.tail.map(_.arg).map(_ match {
            case ColumnRef(None, columnRef) => columnRef
            case _ =>
              throw new RuntimeException(
                "the tail arguments of table/scan/query must be simple identifiers"
              )
          })
          RelationalAlgebra.Schema(args.map(Some(_)))
        }
        Vector(
          OpToken(
            TableOp(
              RelationalAlgebra.AliasTableRef(tableReference),
              None,
              Some(schema)
            )
          )
        )
      case other
          if currentExpressionName.isDefined && other == currentExpressionName.get =>
        throw new RuntimeException(
          s"recursive expression not allowed. ${currentExpressionName.get} refers to itself"
        )
      case other if namedExpressions.contains(other) =>
        require(
          lexicalToken.arguments.isEmpty,
          "referenced expression token must not have an argument list"
        )
        val lifted = liftExpression(
          namedExpressions(other),
          namedExpressions,
          definedFunctions
        )
        lifted

    }
  }

  private def liftExpression(
      expression: SyntaxTree.Expression,
      namedExpressions: Map[String, SyntaxTree.Expression],
      definedFunctions: List[LiftFunction]
  ) = {

    def loop(
        remaining: List[SyntaxTree.Token],
        acc: Vector[StackToken]
    ): Vector[StackToken] =
      remaining match {
        case head :: next =>
          loop(
            next,
            acc ++ resolveToken(
              head,
              namedExpressions,
              expression.boundToName,
              definedFunctions
            )
          )
        case immutable.Nil => acc
      }

    loop(expression.tokenList.toList, Vector.empty)
  }

  private def liftExpressionList(
      parsed: SyntaxTree.ExpressionList,
      definedFunctions: List[LiftFunction]
  ): Either[String, Result] = {
    val tokens = {
      val unnameds = parsed.expressions.filter(_.boundToName.isEmpty)
      if (unnameds.size != 1) Left("Needs exactly 1 anonymous expression")
      else {
        val root = unnameds.head
        val namedExpressions = parsed.expressions
          .filter(_.boundToName.isDefined)
          .map(e => (e.boundToName.get -> e))
        val duplicates = namedExpressions.groupBy(_._1).filter(_._2.size > 1)
        if (duplicates.nonEmpty) {
          throw new RuntimeException(
            s"Duplicate named expressions: ${duplicates.keySet.toSeq.mkString(", ")}"
          )
        }

        val stackTokens =
          liftExpression(root, namedExpressions.toMap, definedFunctions)
        val head = stackTokens.head match {
          case OpToken(op: TableOp) => op
          case _ =>
            throw new RuntimeException(
              "first token in the root expression must be a table/scan/query"
            )

        }
        Right(TokenList(head, stackTokens.toList.drop(1)))
      }
    }
    val schemas = parsed.schemas.map { case SyntaxTree.Schema(name, args) =>
      val liftedArgs = args.map {
        case ColumnRef(None, columnRef) => Option(columnRef)
        case _ =>
          throw new RuntimeException(
            "arguments of schemas must be simple identifiers"
          )
      }
      (
        RelationalAlgebra.AliasTableRef(name),
        RelationalAlgebra.Schema(liftedArgs.toList)
      )
    }

    tokens.map { tokenList =>
      tokenList.compile.bindSchemas(schemas: _*)
    }
  }

  private[tgnn] object SyntaxTree {
    sealed trait Argument

    case class ColumnRef(tableRef: Option[String], columnRef: String)
        extends Argument {
      override def toString =
        tableRef.map(s => s + ".").getOrElse("") + columnRef
    }

    case class VariableRef(name: String) extends Argument {
      override def toString = s"?$name"
    }

    case class FunctionWithArgs(
        name: String,
        args: NonEmptyList[Argument]
    ) extends Argument {
      override def toString = if (
        DslParser.opCharList.contains(name.head) && args.size == 2
      ) s"(${args.head} $name ${args.tail.head})"
      else s"$name(${args.toList.mkString(", ")})"
    }

    case class ArgumentWithAlias(
      arg: Argument,
      alias:  Option[ColumnRef]
    ) {
      override def toString = s"$arg${alias.map(al => s" as $al").getOrElse("")}"
    }
    case class ArgumentList(
        args: NonEmptyList[ArgumentWithAlias]
    ) {
      override def toString = s"${args.toList.mkString(", ")}"
    }

    case class Token(
        name: String,
        arguments: Option[ArgumentList]
    ) {
      override def toString =
        s"$name${arguments.map(args => s"(${args.args.toList.mkString(", ")})").getOrElse("")}"
    }

    sealed trait SchemaOrExpression

    case class Schema(name: String, arguments: NonEmptyList[Argument])
        extends SchemaOrExpression {
      override def toString =
        s"schema $name (${arguments.toList.mkString(", ")})"
    }

    case class Expression(
        boundToName: Option[String],
        tokenList: NonEmptyList[Token]
    ) extends SchemaOrExpression {
      override def toString =
        s"${boundToName.map(s => s"let $s = ").getOrElse("")}${tokenList.toList.mkString(" ")} end"
    }

    case class ExpressionList(
        expressions: NonEmptyList[Expression],
        schemas: List[Schema]
    ) {
      override def toString = (expressions.toList ++ schemas).mkString("\n")
    }
  }

  /** grammar:
    * ~~~
    * program = expressionlist
    * expressionlist = expression [ expressionlist ]
    * expression = ["let" name "=" ] tokenlist
    * name = 'alphanumeric or operator names'
    * tokenlist = token [separator tokenlist]
    * token = name ["(" argumentlist ")"]
    * argumentlist =  argument ["," argumentlist]
    * tableref = name
    * variable = "?" name
    * columnref = [tableref "."] name
    * argument = infix
    * prefixfunction = name "(" argumentlist ")"
    * operand = infix | columnref | variable | prefixfunction | (infix)
    * infix = operand name operand
    * ~~~
    */
  object DslParser {
    import SyntaxTree._
    val underscore = Parser.charIn('_')
    val identifierStart = Rfc5234.alpha | underscore
    val hyphen = Parser.charIn('-')
    val identifierPart = Rfc5234.alpha | Rfc5234.digit

    val identifier =
      (identifierStart ~ (underscore | hyphen | identifierPart).rep0).map {
        case (s, l) => (s :: l).mkString
      }
    val opCharList = "+-:/*&|%<>!=^"
    val opChars = Parser
      .charIn(
        opCharList
      )
    val operatorIdentifier =
      (opChars.rep).map {
        _.toList.mkString
      }

    def operatorStartingWith(s: String) =
      (Parser.charIn(s) ~ opChars.rep0).map { case (x, xs) =>
        (x :: xs).mkString
      }

    val sp = Parser.char(' ')
    val htab = Parser.char('\t')
    val nl = Parser.char('\n')
    val wsp = sp | htab | nl
    val wh = wsp.rep.void
    val optionalWh = wsp.rep0.void
    val period = Parser.char('.')
    val comma = Parser.char(',')
    val open = Parser.char('(')
    val close = Parser.char(')')
    val qmark = Parser.char('?')
    val qualifiedIdentifier = ((identifier.soft <* period).?.with1 ~ identifier)

    val name = identifier
    val operatorName = operatorIdentifier.withContext("operator")
    val variable =
      (qmark *> name).withContext("variable").map(s => VariableRef(s))
    val columnRef = qualifiedIdentifier.withContext("column ref").map {
      case (tableRef, columnRef) =>
        ColumnRef(tableRef, columnRef)
    }

    val precedences = List(
      List('*', '/', '%') -> 0,
      List('+', '-') -> 1,
      List(':') -> 2,
      List('=', '!') -> 3,
      List('<', '>') -> 4,
      List('&') -> 5,
      List('^') -> 6,
      List('|') -> 7
    )

    def higherPrecedenceOperator(c: Char) = {
      val current = precedences
        .find(_._1.contains(c))
        .map(_._2)
        .getOrElse(precedences.map(_._2).max + 1)
      operatorStartingWith(
        precedences.filter(_._2 < current).flatMap(_._1).mkString
      )
    }

    // .map{ case (arg,alias) => ArgumentWithAlias(arg,alias)}
    val asAlias = (Parser.string("as") ~ wh) *> columnRef

    val argumentList = Parser.recursive[NonEmptyList[ArgumentWithAlias]] {
      recursiveArgumentList =>
        val prefixfunction =
          ((name.soft ~ (optionalWh ~ open ~ optionalWh).void) ~ recursiveArgumentList <* optionalWh ~ close)
            .map { case ((name, _), args) =>
              FunctionWithArgs(name, args.map(_.arg))
            }
            .withContext("function")

        val operandAtom =
          prefixfunction.backtrack |
            variable.backtrack |
            columnRef

        val expression = Parser.recursive[Argument] { expression =>
          val simple = operandAtom
          val simpleInParens =
            ((open ~ optionalWh) *> operandAtom <* (optionalWh ~ close))

          val recurseInParens =
            ((open ~ optionalWh) *> expression <* (optionalWh ~ close))

          val operandRecursiveParens =
            simple.backtrack | simpleInParens.backtrack | recurseInParens.backtrack
          val operandRecursive =
            expression | simple.backtrack | simpleInParens.backtrack | recurseInParens.backtrack
          val operandNonRecursive =
            simple.backtrack | simpleInParens.backtrack | recurseInParens.backtrack

          val infix2 =
            ((operandNonRecursive <* optionalWh) ~ operatorName ~ (optionalWh *> operandRecursive))
              .map { case ((arg1, opName), arg2) =>
                FunctionWithArgs(opName, NonEmptyList(arg1, List(arg2)))
              }
          val infix3: Parser[FunctionWithArgs] =
            ((operandNonRecursive <* optionalWh) ~ operatorName <* optionalWh)
              .flatMap { case (arg1, op1) =>
                // arg1 op1 arg2
                val parseHigher =
                  (operandRecursiveParens ~ (optionalWh *> higherPrecedenceOperator(
                    op1.head
                  ))).peek.flatMap { _ =>
                    expression.map { arg2 =>
                      FunctionWithArgs(op1, NonEmptyList(arg1, List(arg2)))
                    }
                  }

                // (arg1 op1 arg2) op2 arg3)
                val parseAny =
                  ((optionalWh *> operandRecursiveParens) ~ (optionalWh *> operatorName) ~ (optionalWh *> operandRecursive))
                    .map { case ((arg2, op2), arg3) =>
                      FunctionWithArgs(
                        op2,
                        NonEmptyList(
                          FunctionWithArgs(
                            op1,
                            NonEmptyList(arg1, List(arg2))
                          ),
                          List(arg3)
                        )
                      )
                    }

                parseHigher.backtrack | parseAny

              }

          (optionalWh.with1 *> (infix3.backtrack | infix2.backtrack | simpleInParens.backtrack | recurseInParens.backtrack | simple) <* optionalWh)

        }

        val argument =
          ((optionalWh.with1 *> expression <* optionalWh) ~ asAlias.backtrack.?).map{ case (a,b) => ArgumentWithAlias(a,b)}

        argument.repSep(comma)

    }

    
    val tableref = name
    val tokenname = name.filter(n => n != "let" && n != "end" && n != "schema")
    val token =
      (tokenname.backtrack ~ ((optionalWh ~ open).backtrack ~ optionalWh *> argumentList  <* optionalWh ~ close).?)
        .map { case (name, args) =>
          Token(
            name,
            args.map { case args =>
              ArgumentList(args)
            }
          )
        }
    val tokenlist = token.repSep(wh) <* (wh ~ Parser.string("end")).backtrack.?
    val letin =
      Parser.string("let") ~ wh *> tokenname <* optionalWh ~ Parser.string("=")
    val schema =
      ((Parser.string(
        "schema"
      ) ~ wh *> identifier <* optionalWh) ~ (open *> argumentList <* close) <* (wh ~ Parser
        .string("end")).backtrack.?).map { case (tableName, schemaArgument) =>
        Schema(tableName, schemaArgument.map(_.arg))
      }
    val expressionWithLet = (((letin <* optionalWh).?).with1 ~ tokenlist)
      .withContext("named expression")
      .map { case (name, tokens) => Expression(name, tokens) }
    val expressionWithoutLet = tokenlist
      .map(s => Option.empty[String] -> s)
      .withContext("anonymous expression")
      .map { case (name, tokens) => Expression(name, tokens) }
    val expression =
      (expressionWithLet.backtrack | schema.backtrack | expressionWithoutLet)
        .withContext("expression")
    val expressionlist =
      Parser.recursive[NonEmptyList[SchemaOrExpression]](recurse =>
        (expression ~ (wh *> recurse <* optionalWh).backtrack.?).map {
          case ((head, tail)) =>
            tail match {
              case Some(tail) => NonEmptyList(head, tail.toList)
              case None       => NonEmptyList(head, Nil)
            }
        }
      )
    val program =
      optionalWh *> expressionlist.map { schemasOrExpressions =>
        val schemas = schemasOrExpressions.collect { case s: Schema =>
          s
        }
        val expressions = schemasOrExpressions.collect { case e: Expression =>
          e
        }
        ExpressionList(
          NonEmptyList(expressions.head, expressions.tail),
          schemas
        )
      } <* optionalWh
  }

}
