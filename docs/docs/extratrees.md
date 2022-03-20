---
title: 'Extratrees'
weight: 5
---

The `lamp-forest` artifact contains an implementation of the Extremely Randomized Trees algorithm.
This implementation is stand alone, it has no dependencies on other lamp modules (neither native dependencies).

Extremely Randomized Trees (extratrees) is a decision tree based regression or classification method similar 
to random forests, ([see](https://hal.archives-ouvertes.fr/hal-00341932)).

### Handling of missing values
Missing feature values (Double.NaNs) are detected and in each split the samples with the missing 
feature values are grouped either to the left or to the right group depending on which one has a better fit. 


### Example

```scala mdoc
import lamp.extratrees._
import org.saddle._

val features = Mat(Vec(1d, 1d, 1d, 1d, 2d,2d,2d))
val target = Vec(1d, 1d, 1d, 1d, 0d, 0d, 0d)
val trees = buildForestRegression(
  data = features, // feature matrix
  target = target, // regression target
  nMin = 1, // minimum node size before splitting
  k = 1, // number of features to consider splitting 
  m = 100, // number of trees
  parallelism = 1
)
val output = predictRegression(trees, features)
```