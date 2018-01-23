## The PARAFAC Model
`PARAFAC` allows the decomposition of three-way data into three loading matrices. 

### Parameters

| Parameter Name | Default Value | Description |
| -------------- | ------------- | ----------- |
| `numComponents` | `3` | Number of components of the loading matrices. |
|`numStarts` | `1` | Number of restarts to find a better minimum. This is only effective if `initMethod=RANDOM`. |
| `initMethod` | `PARAFAC.Initialization.SVD` | Initialization method for the loading matrices. Can be one of `{PARAFAC.Initialization.RANDOM, PARAFAC.Initialization.SVD}`.|

### Example Code 

```java
int nComponents = ... // Choose a number of components F for the loading matrices
double[][][] data = ... // e.g. load data of shape (I x J x K)
Tensor x = Tensor.create(data);
PARAFAC pf = new PARAFAC();
pf.setNumComponents(nComponents);
pf.buildModel(x);
Map<String, Tensor> loads = pf.getLoadingMatrices();
// loads.get("A") is of shape (I x F)
// loads.get("B") is of shape (J x F)
// loads.get("C") is of shape (K x F)
```