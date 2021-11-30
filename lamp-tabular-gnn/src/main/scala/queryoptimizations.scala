package lamp.tgnn

import lamp.tgnn.RelationalAlgebra._
import java.util.UUID
import org.saddle.index.InnerJoin

sealed trait Mutation {
  def makeChildren(parent: Result): Seq[Result]
}
object Mutation {
  val list = List(PushDownFilters, PushDownInnerJoin, PushDownProjection)
}

object PushDownProjection extends Mutation {

  def swap(root: Result, filter: Projection, grandChild: Op): Result = {
    val a = filter
    val b = filter.input
    val c = grandChild
    // before c - b - a
    // after c - a - b
    // b can have other children
    val acopy = a.copy(input = grandChild)
    val bcopy = b.replace(c.id, acopy)
    root.replace(a.id, bcopy).asInstanceOf[Result]
  }

  def trySwap(
      root: Result,
      filter: Projection,
      grandChild: Op,
      provided: Map[UUID, Seq[TableColumnRef]]
  ): Option[Result] = {

    val grandChildSatisfiesDependencies =
      (filter.neededColumns.toSet &~ provided(grandChild.id).toSet).isEmpty

    val childDependencies = filter.input match {
      case x: Op1 => x.neededColumns
      case x: Op2 =>
        if (grandChild.id == x.input1.id) x.neededColumns1 else x.neededColumns2
      case _ => Nil
    }

    val thisNodeSatisfiesChildDependencies =
      (childDependencies.toSet &~ provided(filter.id).toSet).isEmpty

    val childTypeOk = filter.input match {
      case _: Filter | _: Projection | _: Product => true
      case _                                      => false
    }

    if (
      childTypeOk && grandChildSatisfiesDependencies && thisNodeSatisfiesChildDependencies
    )
      Some(swap(root, filter, grandChild))
    else None
  }

  def tryPush(
      root: Result,
      filter: Projection,
      provided: Map[UUID, Seq[TableColumnRef]]
  ): Seq[Result] = {
    val inputInputs = filter.input.inputs
    inputInputs.flatMap { grandChild =>
      trySwap(
        root,
        filter,
        grandChild.op,
        provided
      ).toList
    }
  }

  def makeChildren(parent: Result): Seq[Result] = {
    val sorted = RelationalAlgebra.topologicalSort(parent).reverse
    val provided = providedReferences(sorted, parent.boundTables)
    val eligible = sorted collect { case x: Projection => x }

    eligible.flatMap { filter =>
      tryPush(parent, filter, provided)
    }

  }
}

object PushDownFilters extends Mutation {

  def swap(root: Result, filter: Filter, grandChild: Op): Result = {
    val a = filter
    val b = filter.input
    val c = grandChild
    // before c - b - a
    // after c - a - b
    // b can have other children
    val acopy = a.copy(input = grandChild)
    val bcopy = b.replace(c.id, acopy)
    root.replace(a.id, bcopy).asInstanceOf[Result]
  }

  def trySwap(
      root: Result,
      filter: Filter,
      grandChild: Op,
      provided: Map[UUID, Seq[TableColumnRef]]
  ): Option[Result] = {

    val grandChildSatisfiesDependencies =
      (filter.neededColumns.toSet &~ provided(grandChild.id).toSet).isEmpty

    val childTypeOk = filter.input match {
      case _: Filter | _: Projection | _: Product => true
      case x: EquiJoin if x.joinType == InnerJoin => true
      case _                                      => false
    }

    if (childTypeOk && grandChildSatisfiesDependencies)
      Some(swap(root, filter, grandChild))
    else None
  }

  def tryPushFilter(
      root: Result,
      filter: Filter,
      provided: Map[UUID, Seq[TableColumnRef]]
  ): Seq[Result] = {
    val inputInputs = filter.input.inputs
    inputInputs.flatMap { grandChild =>
      trySwap(
        root,
        filter,
        grandChild.op,
        provided
      ).toList
    }
  }

  def makeChildren(parent: Result): Seq[Result] = {
    val sorted = RelationalAlgebra.topologicalSort(parent).reverse
    val provided = providedReferences(sorted, parent.boundTables)
    val eligibleFilters = sorted collect { case f: Filter =>
      f
    }

    eligibleFilters.flatMap { filter =>
      tryPushFilter(parent, filter, provided)
    }

  }
}
object PushDownInnerJoin extends Mutation {

  def swap(
      root: Result,
      join: EquiJoin,
      child: Op,
      grandChild: Op
  ): Result = {
    val a = join
    val b = child
    val c = grandChild
    // before c - b - a
    // after c - a - b
    // b can have other children
    val acopy =
      if (a.input1.id == child.id) a.copy(input1 = grandChild)
      else a.copy(input2 = grandChild)
    val bcopy = b.replace(c.id, acopy)
    root.replace(a.id, bcopy).asInstanceOf[Result]
  }

  def trySwap(
      root: Result,
      op: EquiJoin,
      child: Op,
      neededColumns: Seq[TableColumnRef],
      grandChild: Op,
      provided: Map[UUID, Seq[TableColumnRef]]
  ): Option[Result] = {

    val grandChildSatisfiesDependencies =
      (neededColumns.toSet &~ provided(grandChild.id).toSet).isEmpty

    val childTypeOk = child match {
      case _: Filter | _: Projection | _: Product => true
      case x: EquiJoin if x.joinType == InnerJoin => true
      case _                                      => false
    }

    if (childTypeOk && grandChildSatisfiesDependencies)
      Some(swap(root, op, child, grandChild))
    else None
  }

  def tryPush(
      root: Result,
      op: EquiJoin,
      provided: Map[UUID, Seq[TableColumnRef]]
  ): Seq[Result] = {
    op.input1.inputs.flatMap { grandChild =>
      trySwap(
        root,
        op,
        op.input1,
        op.neededColumns1,
        grandChild.op,
        provided
      ).toList
    } ++
      op.input2.inputs.flatMap { grandChild =>
        trySwap(
          root,
          op,
          op.input2,
          op.neededColumns2,
          grandChild.op,
          provided
        ).toList
      }
  }

  def makeChildren(root: Result): Seq[Result] = {
    val sorted = topologicalSort(root).reverse
    val provided = providedReferences(sorted, root.boundTables)
    val eligible = sorted collect {
      case f: EquiJoin if f.joinType == InnerJoin =>
        f
    }

    eligible.flatMap { op =>
      tryPush(root, op, provided)
    }

  }
}
