package lamp.tgnn

import cats.parse.Rfc5234
import cats.parse.Parser
import cats.data.NonEmptyList
import scala.collection.immutable
import lamp.tgnn.QueryDsl.SyntaxTree.ColumnRef
import lamp.tgnn.QueryDsl.SyntaxTree.VariableRef
import lamp.tgnn.QueryDsl.SyntaxTree.FunctionWithArgs
import org.saddle.index.InnerJoin
import org.saddle.index.RightJoin
import org.saddle.index.OuterJoin
import org.saddle.index.LeftJoin

object QueryDsl {

  def parse(s: String): Either[Parser.Error, SyntaxTree.ExpressionList] =
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

  def liftColumnRef(cr: ColumnRef): LiftedColumnRef =
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

  def liftArgument(args: SyntaxTree.Argument): LiftedArgument =
    args match {
      case cr: ColumnRef if cr.tableRef.isEmpty && cr.columnRef.toLowerCase == "true"=>
        LiftedBooleanFactor(BooleanAtomTrue)
      case cr: ColumnRef if cr.tableRef.isEmpty && cr.columnRef.toLowerCase == "false"=>
        LiftedBooleanFactor(BooleanAtomFalse)
      case cr: ColumnRef =>
        liftColumnRef(cr)
      case VariableRef(name) =>
        LiftedVariableRef(RelationalAlgebra.VariableRef(name))
      case FunctionWithArgs(name, args) =>
        val liftedArgs = args.map(liftArgument)

        typeAndLiftFunction(name, liftedArgs)

    }

  def typeAndLiftFunction(
      name: String,
      args: NonEmptyList[LiftedArgument]
  ): LiftedArgument =
    name match {
      case "||"
          if args.head.isBooleanFactor && args.size == 2 && args.tail.head.isBooleanFactor =>
        LiftedBooleanFactor(
          args.head.asBooleanFactor.or(args.tail.head.asBooleanFactor)
        )
      case "&&"
          if args.head.isBooleanFactor && args.size == 2 && args.tail.head.isBooleanFactor =>
        LiftedBooleanFactor(
          args.head.asBooleanFactor.and(args.tail.head.asBooleanFactor)
        )
      case "not" if args.head.isBooleanFactor && args.size == 1 =>
        LiftedBooleanFactor(args.head.asBooleanFactor.negate)
      case "identity" | "select" if args.head.isColumnRef =>
        LiftedBooleanFactor(args.head.asColumnRef.select)
      case "==" if args.size == 2 && args.head.isColumnRef =>
        args.tail.head match {
          case LiftedColumnRef(cr2) =>
            LiftedBooleanFactor(args.head.asColumnRef.===(cr2))
          case LiftedVariableRef(vr2) =>
            LiftedBooleanFactor(args.head.asColumnRef.===(vr2))
          case LiftedBooleanFactor(bf) =>
            LiftedBooleanFactor(args.head.asColumnRef.select.===(bf))
        }
      case _ =>
          throw new RuntimeException(s"Implementation for '$name' with args [${args.toList.mkString(", ")}] not found. Operators are left associative without precedence rules. Variables must be the right operand.")

    }

  def resolveToken(
      lexicalToken: SyntaxTree.Token,
      namedExpressions: Map[String, SyntaxTree.Expression],
      currentExpressionName: Option[String]
  ): Vector[StackToken] = {
    lexicalToken.name.toLowerCase match {
      case "filter" =>
        val filterExpression = liftArgument(
          lexicalToken.arguments.get.head
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
        val arguments = lexicalToken.arguments.get.map(arg =>
          liftArgument(arg) match {
            case LiftedColumnRef(
                  cr: RelationalAlgebra.QualifiedTableColumnRef
                ) =>
              cr.select.as(cr)
            case LiftedVariableRef(_) =>
              throw new RuntimeException(
                "project's argument can't be a variable"
              )
            case LiftedBooleanFactor(booleanFactor: ColumnFunction) =>
              booleanFactor.as(Q.table("boo").col("boo"))
            case _ =>
              throw new RuntimeException(
                s"Unexpected type for projection argument $arg"
              )
          }
        )
        Vector(
          StackOp1Token("projection", in => Projection(in, arguments.toList))
        )
      case "inner-join" | "outer-join" | "left-join" | "right-join" =>
        require(
          lexicalToken.arguments.get.size == 2,
          "join needs 2 arguments: join column 1, join column 2"
        )
        val arguments = lexicalToken.arguments.get.map(arg =>
          liftArgument(arg) match {
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
        val variable = lexicalToken.arguments.get.head match {
          case VariableRef(name) => name
          case _ =>
            throw new RuntimeException(
              "table/scan/query expects a single variable argument"
            )
        }
        Vector(
          OpToken(
            TableOp(RelationalAlgebra.AliasTableRef(variable), None)
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
        val lifted = liftExpression(namedExpressions(other), namedExpressions)
        lifted

    }
  }

  def liftExpression(
      expression: SyntaxTree.Expression,
      namedExpressions: Map[String, SyntaxTree.Expression]
  ) = {

    def loop(
        remaining: List[SyntaxTree.Token],
        acc: Vector[StackToken]
    ): Vector[StackToken] =
      remaining match {
        case head :: next =>
          loop(
            next,
            acc ++ resolveToken(head, namedExpressions, expression.boundToName)
          )
        case immutable.Nil => acc
      }

    loop(expression.tokenList.toList, Vector.empty)
  }

  def liftExpressionList(parsed: SyntaxTree.ExpressionList): Either[String, Result] = {
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

        val stackTokens = liftExpression(root, namedExpressions.toMap)
        val head = stackTokens.head match {
          case OpToken(op @ TableOp(_, _)) => op
          case _ =>
            throw new RuntimeException(
              "first token in the root expression must be a table/scan/query"
            )

        }
        Right(TokenList(head, stackTokens.toList.drop(1)))
      }
    }
    tokens.map(_.compile)
  }

  def compile(s:String) = parse(s).left.map(_.toString).flatMap(liftExpressionList)

  object SyntaxTree {
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
      override def toString = if ("+-=&|%/".contains(name.head) && args.size == 2) s"(${args.head} $name ${args.tail.head})"
      else s"$name(${args.toList.mkString(", ")})"
    }

    case class Token(
        name: String,
        arguments: Option[NonEmptyList[Argument]]
    ) {
      override def toString =
        s"$name${arguments.map(arg => s"(${arg.toList.mkString(", ")})").getOrElse("")}"
    }

    case class Expression(
        boundToName: Option[String],
        tokenList: NonEmptyList[Token]
    ) {
      override def toString =
        s"${boundToName.map(s => s"let $s = ").getOrElse("")}${tokenList.toList.mkString(" ")} end"
    }

    case class ExpressionList(expressions: NonEmptyList[Expression]) {
      override def toString = expressions.toList.mkString("\n")
    }
  }
  import SyntaxTree._

  object DslParser {

// program = expressionlist
// expressionlist = expression [ expressionlist ]
// expression = ["let" name "=" ] tokenlist
// name = 'alphanumeric or operator names'
// tokenlist = token [separator tokenlist]
// token = name ["(" argumentlist ")"]
// argumentlist =  argument ["," argumentlist]
// tableref = name
// variable = "?" name
// columnref = [tableref "."] name
// argument = expression
// prefixfunction = name "(" argumentlist ")"
// operand = expression | columnref | variable
// infixfunction = operand name operand

    val identifierStart = Rfc5234.alpha
    val underscore = Parser.charIn('_')
    val hyphen = Parser.charIn('-')
    val identifierPart = Rfc5234.alpha | Rfc5234.digit

    val identifier =
      (identifierStart ~ (underscore | hyphen | identifierPart).rep0).map {
        case (s, l) => (s :: l).mkString
      }
    val opChars = Parser
      .charIn(
        "+-:/*&|%<>!="
      )
    val operatorIdentifier =
      (opChars.rep).map {
        _.toList.mkString
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

    val argumentList = Parser.recursive[NonEmptyList[Argument]] {
      recursiveArgumentList =>
        val prefixfunction =
          ((name.soft ~ (optionalWh ~ open ~ optionalWh).void) ~ recursiveArgumentList <* optionalWh ~ close)
            .map { case ((name, _), args) =>
              FunctionWithArgs(name, args)
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

          val operand0 =
            expression | simple.backtrack | simpleInParens.backtrack | recurseInParens.backtrack
          val operand1 =
            simple.backtrack | simpleInParens.backtrack | recurseInParens.backtrack
          val infix =
            ((operand1 <* optionalWh) ~ operatorName ~ (optionalWh *> operand0))
              .map { case ((arg1, opName), arg2) =>
                FunctionWithArgs(opName, NonEmptyList(arg1, List(arg2)))
              }

          infix.backtrack | simpleInParens.backtrack | recurseInParens.backtrack | simple

        }

        val argument =
          (optionalWh.with1 *> expression <* optionalWh)

        argument.repSep(comma)

    }

    val tableref = name
    val tokenname = name.filter(n => n != "let" && n != "end")
    val token =
      (tokenname.backtrack ~ ((optionalWh ~ open).backtrack ~ optionalWh *> argumentList <* optionalWh ~ close).?)
        .map { case (name, args) => Token(name, args) }
    val tokenlist = token.repSep(wh) <* (wh ~ Parser.string("end")).backtrack.?
    val letin =
      Parser.string("let") ~ wh *> tokenname <* optionalWh ~ Parser.string("=")
    val expressionWithLet = (((letin <* optionalWh).?).with1 ~ tokenlist).withContext("named expression")
    val expressionWithoutLet = tokenlist.map(s => Option.empty[String] -> s).withContext("anonymous expression")
    val expression = (expressionWithLet.backtrack | expressionWithoutLet).map {
      case (letName, tokenList) =>
        Expression(letName, tokenList)
    }.withContext("expression")
    val expressionlist = Parser.recursive[NonEmptyList[Expression]](recurse =>
      (expression ~ (wh *> recurse <* optionalWh).backtrack.?).map { case ((head, tail)) =>
        tail match {
          case Some(tail) => NonEmptyList(head, tail.toList)
          case None       => NonEmptyList(head, Nil)
        }
      }
    )
    val program = optionalWh *> expressionlist.map(ExpressionList(_)) <* optionalWh
  }

}
// val columnfunction =
//   (prefixfunction | infixfunction)
// val booleanExpression = Parser
//   .recursive[BooleanExpression] { recurse =>
//     val booleanFactor: Parser[BooleanFactor] =
//       (open ~ optionalWh *> recurse <* optionalWh.soft ~ close).backtrack |
//         bTrue.map(_ => BTrue) | bFalse.map(_ =>
//           BFalse
//         ) | columnfunction | (not ~ optionalWh *> recurse).map(expr =>
//           BNot(expr)
//         )

//     val booleanTerm =
//       (booleanFactor ~ (wh.soft *> or ~ wh ~ booleanFactor).rep0).map {
//         case (head, tail) =>
//           BooleanTerm(head :: tail.map { case ((_, tail)) => tail })
//       }

//     (booleanTerm ~ (wh.soft *> and ~ wh ~ booleanTerm).rep0).map {
//       case (head, tail) =>
//         BooleanExpression(head :: tail.map { case ((_, tail)) => tail })
//     }
//   }
