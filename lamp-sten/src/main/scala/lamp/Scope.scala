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
import scala.jdk.CollectionConverters._
import java.util.concurrent.ConcurrentLinkedQueue

trait Movable[-R] {
  def list(movable: R): List[Tensor]
}
class EmptyMovable[-R]
object EmptyMovable {
  def make[T] = new EmptyMovable[T]
  implicit def unitIsMovable: EmptyMovable[Unit] = Movable.empty[Unit]
  implicit def stringIsMovable: EmptyMovable[String] = Movable.empty[String]
  implicit def DoubleIsMovable: EmptyMovable[Double] = Movable.empty[Double]
  implicit def FloatIsMovable: EmptyMovable[Float] = Movable.empty[Float]
  implicit def BooleanIsMovable: EmptyMovable[Boolean] = Movable.empty[Boolean]
  implicit def IntIsMovable: EmptyMovable[Int] = Movable.empty[Int]
  implicit def CharIsMovable: EmptyMovable[Char] = Movable.empty[Char]
  implicit def LongIsMovable: EmptyMovable[Long] = Movable.empty[Long]
  implicit def ShortIsMovable: EmptyMovable[Short] = Movable.empty[Short]
  implicit def ByteIsMovable: EmptyMovable[Byte] = Movable.empty[Byte]

  @scala.annotation.nowarn
  implicit def OptionIsMovable[T: EmptyMovable]: EmptyMovable[Option[T]] =
    Movable.empty

  @scala.annotation.nowarn
  implicit def EitherIsMovable[T1: EmptyMovable, T2: EmptyMovable]
      : EmptyMovable[Either[T1, T2]] = Movable.empty

  @scala.annotation.nowarn
  implicit def SeqIsMovable[T: EmptyMovable]: EmptyMovable[Seq[T]] =
    Movable.empty
    
  @scala.annotation.nowarn
  implicit def ArrayIsMovable[T: EmptyMovable]: EmptyMovable[Array[T]] =
    Movable.empty

  @scala.annotation.nowarn
  implicit def t2[T1: EmptyMovable, T2: EmptyMovable]: EmptyMovable[(T1, T2)] =
    Movable.empty

  @scala.annotation.nowarn
  implicit def t3[T1: EmptyMovable, T2: EmptyMovable, T3: EmptyMovable]
      : EmptyMovable[(T1, T2, T3)] =
    Movable.empty

  @scala.annotation.nowarn
  implicit def t4[
      T1: EmptyMovable,
      T2: EmptyMovable,
      T3: EmptyMovable,
      T4: EmptyMovable
  ]: EmptyMovable[(T1, T2, T3, T4)] =
    Movable.empty

  @scala.annotation.nowarn
  implicit def t5[
      T1: EmptyMovable,
      T2: EmptyMovable,
      T3: EmptyMovable,
      T4: EmptyMovable,
      T5: EmptyMovable
  ]: EmptyMovable[(T1, T2, T3, T4, T5)] =
    Movable.empty

  @scala.annotation.nowarn
  implicit def t6[
      T1: EmptyMovable,
      T2: EmptyMovable,
      T3: EmptyMovable,
      T4: EmptyMovable,
      T5: EmptyMovable,
      T6: EmptyMovable
  ]: EmptyMovable[(T1, T2, T3, T4, T5, T6)] = Movable.empty

}
object Movable {
  implicit class MovableSyntax[T: Movable](t: T) {
    def tensors = implicitly[Movable[T]].list(t)
  }
  def empty[T] = new EmptyMovable[T]
  def nonEmpty[T](extract: T => List[Tensor]) = new Movable[T] {
    def list(t: T) = extract(t)
  }
  def by[T, K: Movable](convert: T => K) = new Movable[T] {
    def list(t: T) = convert(t).tensors
  }
  implicit def stensorIsMovable: Movable[STen] = new Movable[STen] {
    def list(m: STen) = List(m.value)
  }

  @scala.annotation.nowarn
  implicit def emptyIsMovable[T](implicit empty: EmptyMovable[T]): Movable[T] =
    new Movable[T] {
      def list(t: T) = Nil
    }

  implicit def OptionIsMovable[T: Movable]: Movable[Option[T]] =
    new Movable[Option[T]] {
      def list(m: Option[T]) =
        m.toList.flatMap(m => implicitly[Movable[T]].list(m)).toList
    }
  implicit def EitherIsMovable[T1: Movable, T2: Movable]
      : Movable[Either[T1, T2]] =
    new Movable[Either[T1, T2]] {
      def list(m: Either[T1, T2]) =
        m.fold(_.tensors, _.tensors)
    }
  implicit def ArrayIsMovable[T: Movable]: Movable[Array[T]] = new Movable[Array[T]] {
    def list(m: Array[T]) = m.flatMap(m => implicitly[Movable[T]].list(m)).toList
  }
  implicit def SeqIsMovable[T: Movable]: Movable[Seq[T]] = new Movable[Seq[T]] {
    def list(m: Seq[T]) = m.flatMap(m => implicitly[Movable[T]].list(m)).toList
  }
  implicit def t2[T1: Movable, T2: Movable]: Movable[(T1, T2)] =
    new Movable[(T1, T2)] {
      def list(m: (T1, T2)) = m._1.tensors ++ m._2.tensors
    }
  implicit def t3[T1: Movable, T2: Movable, T3: Movable]
      : Movable[(T1, T2, T3)] =
    new Movable[(T1, T2, T3)] {
      def list(m: (T1, T2, T3)) = m._1.tensors ++ m._2.tensors ++ m._3.tensors
    }
  implicit def t4[T1: Movable, T2: Movable, T3: Movable, T4: Movable]
      : Movable[(T1, T2, T3, T4)] =
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
  ]: Movable[(T1, T2, T3, T4, T5)] =
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
  ]: Movable[(T1, T2, T3, T4, T5, T6)] =
    new Movable[(T1, T2, T3, T4, T5, T6)] {
      def list(m: (T1, T2, T3, T4, T5, T6)) =
        m._1.tensors ++ m._2.tensors ++ m._3.tensors ++ m._4.tensors ++ m._5.tensors ++ m._6.tensors
    }

}

/** Faciliates memory management of off-heap data structures.
  *
  * Tracks allocations of aten.Tensor and aten.TensorOption instances.
  *
  * aten.Tensor and aten.TensorOption instances are not freed up by the garbage
  * collector. Lamp implements zoned memory management around these object. The
  * managed counterpart of aten.Tensor is [[lamp.STen]], while for
  * aten.TensorOption it is [[lamp.STenOptions]].
  *
  * One can only create a [[lamp.STen]] instance with a [[lamp.Scope]] in
  * implicit scope.
  *
  * Create new scopes with [[lamp.Scope.root]], [[lamp.Scope.apply]] or
  * [[lamp.Scope.root]].
  *
  * =Examples=
  * {{{
  * // Scope.root returns Unit
  * Scope.root { implicit scope =>
  *     val sum = Scope { implicit scope =>
  *     // Intermediate values allocated in this block (`ident` and `ones`) are freed when
  *     // this block returns
  *     // The return value (`ident + ones`) of this block is moved to the outer scope
  *     val ident = STen.eye(3, STenOptions.d)
  *     val ones = STen.ones(List(3, 3), STenOptions.d)
  *     ident + ones
  *     }
  *     assert(sum.toMat == mat.ones(3, 3) + mat.ident(3))
  *     // `sum` is freed once this block exits
  * }
  * }}}
  */
final class Scope private {

  type ResourceType = Either[Tensor, TensorOptions]

  @volatile
  private var closed = false
  @volatile
  private var resources = new ConcurrentLinkedQueue[ResourceType]()

  /** Adds a resource to the managed resources, then returns it unchanged.
    *
    * The resources will be released when this Scope goes out of scope or
    * otherwise releases.
    */
  def apply(resource: Tensor): Tensor = {
    register(resource)
    resource
  }

  /** Adds a resource to the managed resources, then returns it unchanged.
    *
    * The resources will be released when this Scope goes out of scope or
    * otherwise releases.
    */
  def apply(resource: TensorOptions): TensorOptions = {
    register(resource)
    resource
  }

  /** Adds a resource to the managed resources.
    *
    * The resources will be released when this Scope goes out of scope or
    * otherwise releases.
    */
  def register(resource: Tensor): Unit = {
    if (resource == null) throw new NullPointerException("null resource")
    if (closed)
      throw new IllegalStateException("already been closed")
    resources.add(Left(resource)) // Left(resource) :: resources
  }

  /** Adds a resource to the managed resources.
    *
    * The resources will be released when this Scope goes out of scope or
    * otherwise releases.
    */
  def register(resource: TensorOptions): Unit = {
    if (resource == null) throw new NullPointerException("null resource")
    if (closed)
      throw new IllegalStateException("already been closed")
    resources.add(Right(resource))
  }

  /** Immediately release the resources managed by this Scope */
  def release(): Unit = manage[Unit](_ => ())

  private def manageMovable[A](
      op: Scope => A
  )(implicit movable: Movable[A]): (A, List[ResourceType]) = {
    var toThrow: Throwable = null
    var releasable: List[ResourceType] = Nil
    try {
      var last: A = null.asInstanceOf[A]
      try {
        last = op(this)
      } catch {
        case t: Throwable =>
          toThrow = t
      }
      if (last == null) {
        releasable = resources.iterator.asScala.toList
        (last, Nil)
      } else {
        val lastResources = movable.list(last)
        val (movableResources, releasableResources) =
          resources.iterator.asScala.toList.partition(r =>
            r match {
              case Left(r)  => lastResources.exists(v => v eq r)
              case Right(_) => false
            }
          )
        releasable = releasableResources
        (last, movableResources)
      }
    } catch {
      case t: Throwable =>
        toThrow =
          if (toThrow == null) t
          else new RuntimeException(t.getMessage(), toThrow)
        null.asInstanceOf[
          (A, List[ResourceType])
        ] // compiler doesn't know `finally` will throw
    } finally {
      closed = true
      var rs = releasable.distinct
      resources =
        null // allow GC, in case something is holding a reference to `this`
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

    op(this)
      .map { last =>
        val lastResources = movable.list(last)
        val (movables, releasable) =
          resources.iterator.asScala.toList.partition(r =>
            r match {
              case Left(r)  => lastResources.exists(v => v eq r)
              case Right(_) => false
            }
          )
        (Option(last), movables, releasable, Option.empty[Throwable])
      }
      .recover { case t: Throwable =>
        (Option.empty[A], Nil, resources.iterator.asScala.toList, Some(t))
      }
      .flatMap { case (last, movables, releasable, error) =>
        IO {
          closed = true
          var rs = releasable.distinct
          this.resources =
            null // allow GC, in case something is holding a reference to `this`
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
          toThrow
        }.flatMap { toThrow =>
          if (toThrow != null) IO.raiseError(toThrow)
          else if (error.isDefined) IO.raiseError(error.get)
          else IO.pure((last.get, movables))
        }
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
      var rs = resources.iterator.asScala.toList.distinct
      resources =
        null // allow GC, in case something is holding a reference to `this`
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

  /** Create new free standing Scope, not bound to any lexical scope. */
  def free = new Scope

  /** Create new Scope bound to a cats-effect Resource.
    *
    * Will release when the cats-effect Resource cleans up.
    */
  def inResource =
    Resource.make(IO {
      Scope.free
    })(scope => IO { scope.release() })

  def root[A: EmptyMovable](use: Scope => IO[A]): IO[A] =
    Scope.bracket(Scope.free)(use)

  /** Create new Scope bound to a cats-effect IO.
    *
    * Will release when the IO finishes. Return values of the IO are moved to
    * the parent scope.
    */
  def bracket[A: Movable](
      use: Scope => IO[A]
  )(implicit parent: Scope): IO[A] =
    Scope.free.manageMovableIO(use).flatMap { case (last, movables) =>
      IO {
        movables.foreach(a => a.fold(parent.register, parent.register))
        last
      }
    }

  /** Create new Scope bound to a cats-effect IO.
    *
    * Will release when the IO finishes. Return values of the IO are moved to
    * the parent scope.
    */
  def bracket[A: Movable](parent: Scope)(
      use: Scope => IO[A]
  ): IO[A] = {
    implicit val p = parent
    bracket(use)
  }

  /** Create new Scope bound to an anonymous function.
    *
    * Will release when the function returns. Return values of the function are
    * moved to the parent scope. Return values must conform to the
    * [[lamp.Movable]] type class.
    */
  def apply[A: Movable](
      op: Scope => A
  )(implicit parent: Scope): A = {
    val (last, movables) = (new Scope).manageMovable(op)
    if (parent != null) {
      movables.foreach(a => a.fold(parent.register, parent.register))
    }
    last
  }

  /** Create new Scope bound to an anonymous function. Returns nothing.
    *
    * Will release when the function returns.
    */
  @scala.annotation.nowarn
  def root[A: EmptyMovable](
      op: Scope => A
  ): A = {
    (new Scope).manage(op)
  }

  /** Create new Scope bound to an anonymous function. May leak resources.
    *
    * Will release when the function returns. Return values are *not* moved to
    * any parent scope.
    */
  def unsafe[A](
      op: Scope => A
  ): A = {
    (new Scope).manage(op)
  }

}
