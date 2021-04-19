package lamp.autograd

/** Configuration class for the computational graph
  *
  * Automatic mixed precision
  * =========================
  * If this is enabled then lamp downcasts in front of certain operations.
  * The list of operators which will down cast if enabled:
  *   - mm
  *   - bmm
  *   - conv1d, conv2d, conv2dT
  *
  * Operations which are not safe to perform in half precision are casted back to
  * at least single precision. This covers most numeric functions.
  *
  * Operations which are safe to perform in half precision (e.g. manipulating tensor shape) are
  * left as is.
  *
  * This setting has no effect on the storage, allocation or initialization of parameters, neither
  * on the optimizer.
  * If automatic mixed precision is not enabled, then no down-cast is performed.
  * However upcasts from half precision to at least single precision are always inserted if a half
  * precision input is sent into an operation which is not safe in half precision.
  * If automatic is not enabled and there is no half precision input tensor, then no cast is done.
  *
  * @param downCastEnabled If true, then automatic mixed precision graph creation is enabled
  */
case class GraphConfiguration(downCastEnabled: Boolean)
