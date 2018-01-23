# Data Tensors
The library provides a `Tensor` class that can hold data of different orders 
(scalars, vectors, matrices or threeway data). All methods provided by algorithm
apis such as in `SupervisedAlgorithm`, `UnsupervisedAlgorithm`, `Filter`, etc. 
accept `Tensor` objects as method arguments and return `Tensor` objects. 


#### Primitive to Tensor
Tensors can be instantiated with the static `Tensor.create(...)` method, by passing raw `double` data:

- `Tensor.create(double data)`
- `Tensor.create(double[] data)`
- `Tensor.create(double[][] data)`
- `Tensor.create(double[][][] data)`


#### Tensor to primitive

A tensor object `Tensor t` can be converted back into primitive double arrays with:

- `t.toScalar() -> double`
- `t.toArray1d() -> double[]`
- `t.toArray2d() -> double[][]`
- `t.toArray3d() -> double[][][]`

#### Tensor Order
A tensor's order/number of modes can be checked with `t.order()`.

# Reading Data
The `DataReader` class provides methods to read different file formats.

#### Three-Way Sparse Data
`DataReader.read3WaySparse(...)` reads sparse data, giving the indices and the corresponding value, of the following format :
```
x0 y0 z0 value0
x0 y0 z1 value1
...
```

#### Sparse Matrices
`DataReader.readSparseMatrix(...)` reads sparse matrices, giving the indices and the corresponding value, of the following format :
```
x0 y0 value0
x0 y0 value1
...
```
