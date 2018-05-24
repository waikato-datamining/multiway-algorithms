## Non-Negative Tensor Factorization
`Non-Negative Tensor Factorization` is a multi-way extension of standard 
Non-Negative Matrix Factorization.
See also: [Non-negative tensor factorization with applications to statistics and computer vision](https://dl.acm.org/citation.cfm?id=1102451).

### Parameters

| Parameter Name | Default Value | Description |
| -------------- | ------------- | ----------- |
| `numComponents` | `10` | Number of components of the loading matrices. |
| `gradientUpateType` | `GRADIENT_UPDATE_TYPE.NORMALIZED_UPDATE` | Update method for the decomposed values. Can be one of `GRADIENT_UPDATE_TYPE.{NORMALIZED_UPDATE, STEP_UPDATE_CUSTOM,ITERATION_UPDATE_CUSTOM}`. |
| `updater` | `none` | Can be an implementation of `org.nd4j.linalg.learning.config.IUpdater`. Only used of `gradientUpdateType` is not `GRADIENT_UPDATE_TYPE.NORMALIZED_UPDATE` |

### Example Code 

```java
// Get data
Tensor X = Tensor.create(xdata);

// Setup model
int nComponents = ... // Choose a number of components F for the loading matrices
NTF ntf = new NTF();
ntf.setNumComponents(nComponents);

// Build  model
ntf.build(X);
Tensor[] decomposition = ntf.getDecomposition(); // Get #numModes matrices that represent the tensor decomposition
```