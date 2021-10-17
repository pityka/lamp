package lamp.nn.graph

import lamp._
import lamp.autograd._

 case class Graph(
      nodeFeatures: Variable,
      edgeFeatures: Variable,
      edgeI: STen,
      edgeJ: STen,
      vertexPoolingIndices: STen
  )
