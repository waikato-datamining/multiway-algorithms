## The PARAFAC Model
`PARAFAC` allows the decomposition of three-way data into three loading matrices. 

### Parameters

| Parameter Name | Default Value | Description |
| -------------- | ------------- | ----------- |
| `numComponents` | `3` | Number of components of the loading matrices. |
|`numStarts` | `1` | Number of restarts to find a better minimum. This is only effective if `initMethod=RANDOM`. |
| `initMethod` | `PARAFAC.Initialization.SVD` | Initialization method for the loading matrices. Can be one of `{PARAFAC.Initialization.RANDOM, PARAFAC.Initialization.RANDOM_ORTHOGONALIZED, PARAFAC.Initialization.SVD}`.|

### Example Code 

```java
int nComponents = ... // Choose a number of components F for the loading matrices
double[][][] data = ... // e.g. load data of shape (I x J x K)
Tensor X = Tensor.create(data);
PARAFAC pf = new PARAFAC();
pf.setNumComponents(nComponents);
pf.build(X);
Map<String, Tensor> loads = pf.getLoadingMatrices();
// loads.get("A") is of shape (I x F)
// loads.get("B") is of shape (J x F)
// loads.get("C") is of shape (K x F)
```

### Generate Scores using Loadings of a Calibrated Model
It is possible to generate scores for new data having the same dimension in the second and third mode based on an already calibrated PARAFAC model. See also: [Multi‐way prediction in the presence of uncalibrated interferents](https://onlinelibrary.wiley.com/doi/full/10.1002/cem.1037).

```java
// Data
Tensor Xtrain = ... // Data of shape (I_1 x J x K)
Tensor Xnew = ... // Data of shape (I_2 x J x K)

// Model setup
PARAFAC pf = new PARAFAC()
pf.setNumComponents(F);
...

// Generate new scores of shape (I_2 x F)
Tensor scores = pf.filter(Xnew);

```