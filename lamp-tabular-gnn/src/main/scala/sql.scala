package lamp.tgnn

import cats.parse.Rfc5234
import cats.parse.Parser
import cats.parse.Numbers

object Sql {

  // val kwSelect = Parser.string("select")
  val identifierStart = Rfc5234.alpha
  val underscore = Parser.charIn('_')
  val identifierPart = Rfc5234.alpha | Rfc5234.digit

  val identifier = (identifierStart ~ (underscore | identifierPart ).rep0).map{ case (s,l) => (s::l).mkString} 

  val period = Parser.char('.')
  val quote = Parser.char('\'')
  val asterisk = Parser.char('*')
  val comma = Parser.char(',')
  val qualifiedIdentifier = (identifier ~ period ~ identifier).map{ case ((q,_),i) => (q,i)}

  val numericLiteral = Numbers.jsonNumber
  val stringLiteral = (quote *> Parser.anyChar.rep0 <* quote).map(_.mkString)
  val literal = numericLiteral | stringLiteral

  val distinct = Parser.string("distinct").rep0(0,1).map(_.size > 0)

  val columnReference = identifier | qualifiedIdentifier
  val asClause = Parser.string("as") *> identifier

  val wh = Rfc5234.wsp.rep

  val valueExpression = columnReference
  
  val derivedColumn = valueExpression ~ asClause.rep0
  val selectSublist = identifier ~ period ~ asterisk | derivedColumn

  val selectList = asterisk | selectSublist.rep

  // va

  val fromClause =   Parser.string("from") *> wh *> identifier.repSep(comma.surroundedBy(wh))

  val joinType = Parser.string("inner") | Parser.string("left") | Parser.string("right") | Parser.string("full") 

  val or = Parser.string("or")
  val and = Parser.string("and")
  val not = Parser.string("not")
  val is = Parser.string("is")
  val bTrue = Parser.string("true")
  val bFalse = Parser.string("false")
  val bNull = Parser.string("null")
  val equalsOp = Parser.string("=")
  val booleanValue = bTrue | bFalse | bNull

  trait BooleanAST

  val comparisonOperator = equalsOp 
  val comparisonPredicate = columnReference ~ wh ~ comparisonOperator ~ wh ~ columnReference
  val predicate = comparisonPredicate 
  val booleanPrimary = predicate 
  val isnot = is ~ wh ~ not.rep0 ~ wh ~ booleanValue
  val booleanTest = booleanPrimary ~ isnot.rep0
  val booleanFactor  : Parser[BooleanAST] = (not.rep0.with1 ~ wh ~ booleanTest).map(_.asInstanceOf[BooleanAST])
  val booleanTerm : Parser[BooleanAST] = Parser.recursive[BooleanAST]( recurse => booleanFactor | (recurse ~ wh ~ and ~ wh ~ booleanFactor).map(_.asInstanceOf[BooleanAST]))
  val booleanExpression = Parser.recursive[BooleanAST]( recurse => booleanTerm | (recurse ~ wh ~ or ~ wh ~ booleanTerm).map(_ => ???))

  val joinedTable = joinType.rep0.with1 ~ wh ~ Parser.string("join") ~ wh ~ identifier ~ wh ~ Parser.string("on") ~ wh ~ booleanExpression

  val tableExpression = fromClause ~ wh ~ joinedTable.rep0 //~ whereClause.rep0 ~ groupByClause.rep0 ~ havingClause.rep0

  val query = Parser.string("select") *> wh *> distinct ~ wh ~ selectList ~ wh ~ tableExpression
  
  

}