# multiway-algorithms

Java library of multi-way algorithms.


## Algorithms

Available algorithms:

* [PARAFAC](http://www.models.life.ku.dk/~rasmus/presentations/parafac_tutorial/paraf.htm)

Planned:

* [Multi-way PLS](http://www.models.life.ku.dk/~rasmus/presentations/Npls_sugar/npls.htm)
* [Non-negative Matrix Factorization (NMF)](https://www.csie.ntu.edu.tw/~cjlin/nmf/)
* [Multivariate Filtering](http://wiki.eigenvector.com/index.php?title=Advanced_Preprocessing:_Multivariate_Filtering)


## Usage

### Stopping Criteria for Iterative Algorithms
Iterative algorithms allow for different stopping criteria such as:

- `TimeStoppingCriterion`: Define a maximum time in seconds
- `ImprovementStoppingCriterion`: Stop after improvement between two iterations is below a certain threshold
- `IterationStoppingCriterion`: Stop after `maxIter` number of iterations

Multiple criteria can be added as follows:
```java
PARAFAC alg = new PARAFAC(...);
alg.addStoppingCriterion(new IterationStoppingCriterion(1000)); // Stop after 1000 iterations
alg.addStoppingCriterion(new TimeStoppingCriterion(100)); // Stop after 100 seconds
alg.addStoppingCriterion(new ImprovementStoppingCriterion(10E-10)); // Stop if relative improvement is less than 10E-10
```
If one of the criteria matches, the algorithm stops. Adding a certain criterion multiple times will result in overwriting.

### Algorithms
#### PARAFAC (three-way)
`PARAFAC` allows the decomposition of three-way data into three loading matrices. 
Example:
```java
int nComponents = ... // Choose a number of components F for the loading matrices
double[][][] data = ... // e.g. load data of shape (I x J x K)
PARAFAC pf = new PARAFAC(nComponents);
pf.buildModel(data);
double[][][] loadingMatrices = pf.getLoadingMatrices();
// loadingMatrices[0] is of shape (I x F)
// loadingMatrices[1] is of shape (J x F)
// loadingMatrices[2] is of shape (K x F)
```

The loading matrices can further be initialized either randomly drawing from `N(0,1)` or with the eigenvectors of the matricized data tensor along each axis. `PARAFAC.Initialization` can be passed to the `PARAFAC` constructor.