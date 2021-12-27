package lamp.tgnn

import cats.parse.Rfc5234
import cats.parse.Parser
import cats.data.NonEmptyList

object QueryDsl {

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
  val identifierPart = Rfc5234.alpha | Rfc5234.digit

  val identifier =
    (identifierStart ~ (underscore.backtrack | identifierPart).rep0).map {
      case (s, l) => (s :: l).mkString
    }
  val opChars = Parser
    .charIn(
      "+-:/*&|%<>!"
    )
  val operatoridentifier =
    (opChars.rep ).map {
      _.toList.mkString
    }

  val wh = Rfc5234.wsp.rep.void
  val optionalWh = Rfc5234.wsp.rep0.void
  val period = Parser.char('.')
  val quote = Parser.char('\'')
  val asterisk = Parser.char('*')
  val comma = Parser.char(',')
  val open = Parser.char('(')
  val close = Parser.char(')')
  val qualifiedIdentifier = ((identifier.soft <* period).?.with1 ~ identifier)

  val qmark = Parser.string("?")

  val name = identifier
  val operatorname = operatoridentifier
  val variable = (qmark *> name).map(s => VariableRef(s))
  val columnref = qualifiedIdentifier.map { case (tableRef, columnRef) =>
    ColumnRef(tableRef, columnRef)
  }

  val argumentlist = Parser.recursive[NonEmptyList[Argument]] { recurse =>
    val prefixfunction =
      ((name.soft ~ (optionalWh ~ open ~ optionalWh).void) ~ recurse <* optionalWh ~ close)
        .map { case ((name, _), args) =>
          FunctionWithArgs(name, args)
        }

    val operandAtom =
      prefixfunction.withContext("pf").backtrack |
        variable.withContext("var").backtrack |
        columnref.withContext("cf")

    val expression = Parser.recursive[Argument] { expression =>
      val simple = operandAtom.withContext("A")
      val simpleInParens =
        ((open ~ optionalWh) *> operandAtom <* (optionalWh ~ close))
          .withContext("B")
      val recurseInParens =
        ((open ~ optionalWh) *> expression <* (optionalWh ~ close))
          .withContext("C")
      val operand0 =
        expression | simple.backtrack | simpleInParens.backtrack | recurseInParens.backtrack
      val operand1 =
        simple.backtrack | simpleInParens.backtrack | recurseInParens.backtrack
      val infix =
        ((operand1 <* optionalWh) ~ operatorname ~ (optionalWh *> operand0)).map{
          case ((arg1,opName),arg2) => FunctionWithArgs(opName,NonEmptyList(arg1,List(arg2)))
        }

      infix.backtrack | simpleInParens.backtrack | recurseInParens.backtrack | simple

    }

    val argument =
      (optionalWh.with1 ~ expression ~ optionalWh)
        .map { case ((_, x), _) =>
          x
        }
    val argumentlist = argument.repSep(comma)

    argumentlist
  }

  val tableref = name
  val tokenname = name.filter(n => n != "let" && n != "end")
  val token =
    (tokenname.backtrack ~ ((optionalWh ~ open).backtrack ~ optionalWh *> argumentlist <* optionalWh ~ close).?)
      .map { case (name, args) => Token(name, args) }
  val tokenlist = token.repSep(wh) <* (wh ~ Parser.string("end")).backtrack.?
  val letin =
    Parser.string("let") ~ wh *> tokenname <* optionalWh ~ Parser.string("=")
  val expressionWithLet = ((letin <* optionalWh).?).with1 ~ tokenlist
  val expressionWithoutLet = tokenlist.map(s => Option.empty[String] -> s)
  val expression = (expressionWithLet.backtrack | expressionWithoutLet).map {
    case (letName, tokenList) =>
      Expression(letName, tokenList)
  }
  val expressionlist = Parser.recursive[NonEmptyList[Expression]](recurse =>
    (expression ~ (wh *> recurse).?).map { case ((head, tail)) =>
      tail match {
        case Some(tail) => NonEmptyList(head, tail.toList)
        case None       => NonEmptyList(head, Nil)
      }
    }
  )
  val program = expressionlist


sealed trait Argument 



  case class ColumnRef(tableRef: Option[String], columnRef: String) extends Argument
      
  case class VariableRef(name: String) extends Argument


  case class FunctionWithArgs(
      name: String,
      args: NonEmptyList[Argument]
  ) extends Argument

  case class Token(
      name: String,
      arguments: Option[NonEmptyList[Argument]]
  )

  case class Expression(
      boundToName: Option[String],
      tokenList: NonEmptyList[Token]
  )

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
