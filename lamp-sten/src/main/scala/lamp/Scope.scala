/* This file contains code derived from
 * https://github.com/scala/scala/blob/2.13.x/src/library/scala/util/Using.scala
 *
 * Using.scala is distributed with the following copyright header:
 * Copyright EPFL and Lightbend, Inc.
 *
 * Licensed under Apache License 2.0
 * (http://www.apache.org/licenses/LICENSE-2.0).
 *
 * See the NOTICE file distributed with this work for
 * additional information regarding copyright ownership.
 *
 * The content of the above referred NOTICE file:
 * ```
 * Scala
 * Copyright (c) 2002-2020 EPFL
 * Copyright (c) 2011-2020 Lightbend, Inc.
 *
 * Scala includes software developed at
 * LAMP/EPFL (https://lamp.epfl.ch/) and
 * Lightbend, Inc. (https://www.lightbend.com/).
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This software includes projects with other licenses -- see `doc/LICENSE.md`.
 * ```
 *
 * Changes:
 * - rename package
 * - rename to Scope and keep only that feature
 * - remove some comments
 * - remove Try
 * - remove preferential suppress
 * - Add Movable
 */

package lamp

import aten.Tensor
import cats.effect.Resource
import cats.effect.IO
import aten.TensorOptions

trait Movable[-R] {
  def list(movable: R): List[Tensor]
}
object Movable {
  implicit class MovableSyntax[T: Movable](t: T) {
    def tensors = implicitly[Movable[T]].list(t)
  }
  def empty[T] = new Movable[T] {
    def list(t: T) = Nil
  }
  def nonEmpty[T](extract: T => List[Tensor]) = new Movable[T] {
    def list(t: T) = extract(t)
  }
  def by[T, K: Movable](convert: T => K) = new Movable[T] {
    def list(t: T) = convert(t).tensors
  }
  implicit def stensorIsMovable = new Movable[STen] {
    def list(m: STen) = List(m.value)
  }

  implicit def unitIsMovable = Movable.empty[Unit]
  implicit def stringIsMovable = Movable.empty[String]
  implicit def DoubleIsMovable = Movable.empty[Double]
  implicit def FloatIsMovable = Movable.empty[Float]
  implicit def BooleanIsMovable = Movable.empty[Boolean]
  implicit def IntIsMovable = Movable.empty[Int]
  implicit def LongIsMovable = Movable.empty[Long]
  implicit def ShortIsMovable = Movable.empty[Short]
  implicit def ByteIsMovable = Movable.empty[Byte]
  implicit def MatDoubleIsMovable = Movable.empty[org.saddle.Mat[Double]]
  implicit def VecDoubleIsMovable = Movable.empty[org.saddle.Vec[Double]]
  implicit def MatIntIsMovable = Movable.empty[org.saddle.Mat[Int]]
  implicit def VecIntIsMovable = Movable.empty[org.saddle.Vec[Int]]
  implicit def OptionIsMovable[T: Movable] = new Movable[Option[T]] {
    def list(m: Option[T]) =
      m.toList.flatMap(m => implicitly[Movable[T]].list(m)).toList
  }
  implicit def EitherIsMovable[T1: Movable, T2: Movable] =
    new Movable[Either[T1, T2]] {
      def list(m: Either[T1, T2]) =
        m.fold(_.tensors, _.tensors)
    }
  implicit def SeqIsMovable[T: Movable] = new Movable[Seq[T]] {
    def list(m: Seq[T]) = m.flatMap(m => implicitly[Movable[T]].list(m)).toList
  }
  implicit def t2[T1: Movable, T2: Movable] = new Movable[(T1, T2)] {
    def list(m: (T1, T2)) = m._1.tensors ++ m._2.tensors
  }
  implicit def t3[T1: Movable, T2: Movable, T3: Movable] =
    new Movable[(T1, T2, T3)] {
      def list(m: (T1, T2, T3)) = m._1.tensors ++ m._2.tensors ++ m._3.tensors
    }
  implicit def t4[T1: Movable, T2: Movable, T3: Movable, T4: Movable] =
    new Movable[(T1, T2, T3, T4)] {
      def list(m: (T1, T2, T3, T4)) =
        m._1.tensors ++ m._2.tensors ++ m._3.tensors ++ m._4.tensors
    }
  implicit def t5[
      T1: Movable,
      T2: Movable,
      T3: Movable,
      T4: Movable,
      T5: Movable
  ] =
    new Movable[(T1, T2, T3, T4, T5)] {
      def list(m: (T1, T2, T3, T4, T5)) =
        m._1.tensors ++ m._2.tensors ++ m._3.tensors ++ m._4.tensors ++ m._5.tensors
    }
  implicit def t6[
      T1: Movable,
      T2: Movable,
      T3: Movable,
      T4: Movable,
      T5: Movable,
      T6: Movable
  ] =
    new Movable[(T1, T2, T3, T4, T5, T6)] {
      def list(m: (T1, T2, T3, T4, T5, T6)) =
        m._1.tensors ++ m._2.tensors ++ m._3.tensors ++ m._4.tensors ++ m._5.tensors ++ m._6.tensors
    }

}

final class Scope private {

  type ResourceType = Either[Tensor, TensorOptions]

  private var closed = false
  private var resources: List[ResourceType] = Nil

  def apply(resource: Tensor): Tensor = {
    register(resource)
    resource
  }
  def apply(resource: TensorOptions): TensorOptions = {
    register(resource)
    resource
  }

  def register(resource: Tensor): Unit = {
    if (resource == null) throw new NullPointerException("null resource")
    if (closed)
      throw new IllegalStateException("already been closed")
    resources = Left(resource) :: resources
  }
  def register(resource: TensorOptions): Unit = {
    if (resource == null) throw new NullPointerException("null resource")
    if (closed)
      throw new IllegalStateException("already been closed")
    resources = Right(resource) :: resources
  }

  def release(): Unit = manage[Unit](_ => ())

  private def manageMovable[A](
      op: Scope => A
  )(implicit movable: Movable[A]): (A, List[ResourceType]) = {
    var toThrow: Throwable = null
    try {
      val last = op(this)
      val lastResources = movable.list(last)
      val (movableResources, releasableResources) =
        resources.partition(r =>
          r match {
            case Left(r)  => lastResources.exists(v => v eq r)
            case Right(_) => false
          }
        )
      resources = releasableResources
      (last, movableResources)
    } catch {
      case t: Throwable =>
        toThrow = t
        null.asInstanceOf[
          (A, List[ResourceType])
        ] // compiler doesn't know `finally` will throw
    } finally {
      closed = true
      var rs = resources.distinct
      resources = null // allow GC, in case something is holding a reference to `this`
      while (rs.nonEmpty) {
        val resource = rs.head
        rs = rs.tail
        try resource.fold(_.release(), _.release())
        catch {
          case t: Throwable =>
            if (toThrow == null) toThrow = t
            else toThrow = t
        }
      }
      if (toThrow != null) throw toThrow
    }
  }
  private def manageMovableIO[A](
      op: Scope => IO[A]
  )(implicit movable: Movable[A]): IO[(A, List[ResourceType])] = {
    import cats.syntax.all._

    op(this)
      .map { last =>
        val lastResources = movable.list(last)
        val (movables, releasable) =
          resources.partition(r =>
            r match {
              case Left(r)  => lastResources.exists(v => v eq r)
              case Right(_) => false
            }
          )
        (Option(last), movables, releasable, Option.empty[Throwable])
      }
      .recover {
        case t: Throwable => (Option.empty[A], Nil, resources, Some(t))
      }
      .flatMap {
        case (last, movables, releasable, error) =>
          closed = true
          var rs = releasable.distinct
          this.resources = null // allow GC, in case something is holding a reference to `this`
          var toThrow: Throwable = null
          while (rs.nonEmpty) {
            val resource = rs.head
            rs = rs.tail
            try resource.fold(_.release(), _.release())
            catch {
              case t: Throwable =>
                toThrow = t
            }
          }
          if (toThrow != null) IO.raiseError(toThrow)
          else if (error.isDefined) IO.raiseError(error.get)
          else IO.pure((last.get, movables))
      }

  }
  private def manage[A](
      op: Scope => A
  ): A = {
    var toThrow: Throwable = null
    try {
      op(this)
    } catch {
      case t: Throwable =>
        toThrow = t
        null.asInstanceOf[A] // compiler doesn't know `finally` will throw
    } finally {
      closed = true
      var rs = resources.distinct
      resources = null // allow GC, in case something is holding a reference to `this`
      while (rs.nonEmpty) {
        val resource = rs.head
        rs = rs.tail
        try resource.fold(_.release(), _.release())
        catch {
          case t: Throwable =>
            if (toThrow == null) toThrow = t
            else toThrow = t
        }
      }
      if (toThrow != null) throw toThrow
    }
  }
}

object Scope {

  def free = new Scope

  def inResource =
    Resource.make(IO {
      Scope.free
    })(scope => IO { scope.release })

  def bracket[A: Movable](
      use: Scope => IO[A]
  )(implicit parent: Scope): IO[A] =
    Scope.free.manageMovableIO(use).flatMap {
      case (last, movables) =>
        IO {
          movables.foreach(a => a.fold(parent.register, parent.register))
          last
        }
    }

  def bracket[A: Movable](parent: Scope)(
      use: Scope => IO[A]
  ): IO[A] = {
    implicit val p = parent
    bracket(use)
  }

  def apply[A: Movable](
      op: Scope => A
  )(implicit parent: Scope): A = {
    val (last, movables) = (new Scope).manageMovable(op)
    if (parent != null) {
      movables.foreach(a => a.fold(parent.register, parent.register))
    }
    last
  }
  def root[A](
      op: Scope => Unit
  ): Unit = {
    (new Scope).manage(op)
  }
  def leak[A](
      op: Scope => A
  ): A = {
    (new Scope).manage(op)
  }

}
