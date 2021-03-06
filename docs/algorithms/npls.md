## Multilinear Partial Least Squares
`Multilinear Partial Least Squares` is a multi-way extension of standard PLS.
See also: [Multiway calibration. Multilinear PLS, Bro 96](http://onlinelibrary.wiley.com/doi/10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C/full).

This implementation extends the PLS2 algorithm to threeway input and thus works on multitarget `Y` data.

### Parameters

| Parameter Name | Default Value | Description |
| -------------- | ------------- | ----------- |
| `numComponents` | `10` | Number of components of the loading matrices. |
| `standardizeY` | `true` | Whether to standardize the `Y` target matrix |

### Example Code 

```java
// Get data
double[][][] xtrain = ... // e.g. load data of shape (I_train x J x K)
double[][][] xtest = ... // e.g. load data of shape (I_test x J x K)
double[][] ytrain = ... // e.g. load data of shape (I_train x M)
double[][] ytest = ... // e.g. load data of shape (I_test x M)
Tensor Xtr = Tensor.create(xtrain);
Tensor Xte = Tensor.create(xtest);
Tensor Ytr = Tensor.create(ytrain);
Tensor Yte = Tensor.create(ytest);

// Setup model
int nComponents = ... // Choose a number of components F for the loading matrices
MultiLinearPLS npls = new MultiLinearPLS();
npls.setNumComponents(nComponents);

// Build and test model
npls.build(Xtr, Ytr);
Tensor Ypred = npls.predict(Yte);
double mse = MathUtils.meanSquaredError(Yte, Ypred);

// Usage as a filter/feature-transformation
Tensor transformedXte = npls.filter(Xte)
```