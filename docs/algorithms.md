#### PARAFAC (three-way)
`PARAFAC` allows the decomposition of three-way data into three loading matrices. 
Example:
```java
int nComponents = ... // Choose a number of components F for the loading matrices
double[][][] data = ... // e.g. load data of shape (I x J x K)
PARAFAC pf = new PARAFAC();
pf.setNumComponents(nComponents);
pf.buildModel(data);
double[][][] loadingMatrices = pf.getBestLoadingMatrices();
// loadingMatrices[0] is of shape (I x F)
// loadingMatrices[1] is of shape (J x F)
// loadingMatrices[2] is of shape (K x F)
```

The loading matrices can further be initialized either randomly drawing from `N(0,1)` or with the eigenvectors of the matricized data tensor along each axis. `PARAFAC.Initialization` can be passed to the `PARAFAC` constructor.