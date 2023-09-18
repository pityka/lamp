package lamp.pretrained

import lamp._
import java.io.File

package object bert {

  

  def allocateBertEncoder(weightsFile:File)(implicit scope: Scope) = {
    val weightTensors = lamp.data.safetensors.SafeTensorReader.read(weightsFile,CPU)
    println(weightTensors.meta)
    println(weightTensors.tensors.keySet.mkString("\n"))
  }

}