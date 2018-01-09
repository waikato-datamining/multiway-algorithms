# Stopping Iterative Algorithms
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
**Note**:
- Algorithm stops if one of the criteria matches 
- Adding a certain criterion multiple times will result in overwriting.
