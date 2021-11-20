package lamp.tgnn

import cats.parse.Rfc5234
import cats.parse.Parser
import cats.parse.Numbers

object Sql {

  // val kwSelect = Parser.string("select")
  val identifierStart = Rfc5234.alpha
  val underscore = Parser.charIn('_')
  val identifierPart = Rfc5234.alpha | Rfc5234.digit

  val identifier =
    (identifierStart ~ (underscore.backtrack | identifierPart).rep0).map {
      case (s, l) => (s :: l).mkString
    }

  val wh = Rfc5234.wsp.rep
  val optionalWh = Rfc5234.wsp.rep0
  val period = Parser.char('.')
  val quote = Parser.char('\'')
  val asterisk = Parser.char('*')
  val comma = Parser.char(',')
  val open = Parser.char('(')
  val close = Parser.char(')')
  val qualifiedIdentifier = ((identifier.soft <* period).?.with1 ~ identifier)

  val numericLiteral = Numbers.jsonNumber
  val stringLiteral = (quote *> Parser.anyChar.rep0 <* quote).map(_.mkString)
  val literal = numericLiteral | stringLiteral

  val distinct = (Parser.string("distinct") <* wh).?.map(_.size > 0)

  val columnReference = qualifiedIdentifier
  val asClause = Parser.string("as") *> identifier

  val valueExpression = columnReference

  val derivedColumn = valueExpression ~ asClause.rep0
  val selectSublist = identifier ~ period ~ asterisk | derivedColumn

  val selectList = asterisk | selectSublist.rep

  // va

  val fromClause = Parser.string("from") *> wh *> identifier.repSep(
    comma.soft.surroundedBy(optionalWh)
  )

  val joinType = Parser.string("inner") | Parser.string("left") | Parser.string(
    "right"
  ) | Parser.string("full")

  val or = Parser.string("or")
  val and = Parser.string("and")
  val not = Parser.string("not")
  val is = Parser.string("is")
  val bTrue = Parser.string("true")
  val bFalse = Parser.string("false")
  val bNull = Parser.string("null")
  val equalsOp = Parser.string("=")
  val booleanValue = bTrue | bFalse | bNull

  class BooleanAST

  val comparisonOperator = equalsOp
  val comparisonPredicate =
    columnReference ~ wh ~ comparisonOperator ~ wh ~ columnReference
  val predicate = comparisonPredicate
  val booleanPrimary = predicate
  val booleanTest = booleanPrimary

  val booleanExpression: Parser[BooleanAST] = Parser.recursive[BooleanAST] {
    recurse =>
      val booleanFactor =
        (open ~ optionalWh *> recurse <* optionalWh.soft ~ close).backtrack  |
        booleanTest

        // (not ~ wh).void.?.map(
        //   _.isDefined
        // ).with1 ~ 
        
      val booleanTerm = booleanFactor ~ (wh.soft *> and <* wh ~ booleanFactor).rep0

      (booleanTerm ~ (wh.soft *> or <* wh ~ booleanTerm).rep0).map(_ => new BooleanAST)
  }
  

  val joinedTable =
    (joinType ~ wh).? ~ Parser.string("join") ~ wh ~ identifier ~ wh ~ Parser
      .string("on") ~ wh ~ booleanExpression

  val tableExpression =
    fromClause ~ (wh ~ joinedTable).? //~ whereClause.rep0 ~ groupByClause.rep0 ~ havingClause.rep0

  val query = Parser.string(
    "select"
  ) *> wh *> distinct ~ selectList ~ wh ~ tableExpression

}
