// package lamp.nn

// import lamp.autograd.{const, param, Variable}
// import aten.Tensor

// object Mod2 {

//   case class SimpleState(state: Seq[(Variable, PTag)], training: Boolean) {

//     def load(parameters: Seq[Tensor]): SimpleState = {
//       val v = parameters.zip(state).map {
//         case (tensor, (current, tag)) =>
//           (if (current.needsGrad) param(tensor) else const(tensor), tag)
//       }
//       copy(state = v)
//     }
//     def release() = state.foreach(_._1.value.release)

//   }

//   case class Simple[A, B](
//       simpleState: SimpleState,
//       fun: SimpleState => A => B
//   ) extends Module[A, B] {
//     def forward(a: A): B = fun(simpleState)(a)
//     def asEval = copy(simpleState = simpleState.copy(training = false))
//     def asTraining = copy(simpleState = simpleState.copy(training = true))
//     def load(parameters: Seq[Tensor]) =
//       copy(simpleState = simpleState.load(parameters))
//     def release() = simpleState.release
//     def state = simpleState.state
//   }

//   case class Fun[A, B](fun: A => B) extends Module[A, B] {

//     override def asEval: Module[A, B] = this

//     override def asTraining: Module[A, B] = this

//     override def load(parameters: Seq[Tensor]): Module[A, B] = Nil

//     override def release(): Unit = ()

//     override def state: Seq[(Variable, PTag)] = Nil

//     def forward(x: A): B = fun(x)
//   }

//   object Sequential {

//     case class Tag[T <: PTag](t: T, idx: Int) extends PTag {
//       def leaf = t
//       def updateDuringOptimization: Boolean = t.updateDuringOptimization
//     }
//   }

//   case class Sequential[A](members: Module[A, A]*) extends Module[A, A] {
//     override def asEval: Sequential[A] =
//       Sequential(members.map(_.asEval): _*)
//     override def asTraining: Sequential[A] =
//       Sequential(members.map(_.asTraining): _*)
//     override def state =
//       members.zipWithIndex.flatMap {
//         case (member, idx) =>
//           member.state.map {
//             case (param, ptag) => (param, Sequential.Tag(ptag, idx))
//           }
//       }
//     def forward(x: A) =
//       members.foldLeft(x) {
//         case (x, m) =>
//           m.forward(x)
//       }
//     def release = members.foreach(_.release)
//     def load(tensors: Seq[Tensor]) = {
//       val (loadedMembers, _) =
//         members.foldLeft((List[Module[A, A]](), tensors)) {
//           case ((acc, params), member) =>
//             val numParam = member.state.size
//             val loaded = member.load(params.take(numParam))
//             (acc.:+(loaded), params.drop(numParam))

//         }
//       Sequential(loadedMembers: _*)
//     }
//   }

//   trait Module[A, B] {
//     def forward(a: A): B
//     def asEval: Module[A, B]
//     def asTraining: Module[A, B]
//     def load(parameters: Seq[Tensor]): Module[A, B]
//     def release(): Unit

//     def state: Seq[(Variable, PTag)]
//     final def parameters =
//       state.filter(v =>
//         v._1.needsGrad && v._1.leaf && v._2.updateDuringOptimization
//       )
//     final def gradients(
//         loss: Variable,
//         zeroGrad: Boolean = true
//     ): Seq[Option[Tensor]] = {
//       if (zeroGrad) {
//         parameters.foreach {
//           case (param, tag) =>
//             param.zeroGrad()
//         }
//       }
//       loss.backprop()
//       val g = parameters.map {
//         case (param, _) => param.partialDerivative
//       }
//       loss.releaseAll
//       g
//     }

//   }

// }
