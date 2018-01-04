package com.github.waikatodatamining.multiway.algorithm;

import com.github.waikatodatamining.multiway.algorithm.stopping.CriterionType;
import com.github.waikatodatamining.multiway.algorithm.stopping.ImprovementStoppingCriterion;
import com.github.waikatodatamining.multiway.algorithm.stopping.IterationStoppingCriterion;
import com.github.waikatodatamining.multiway.algorithm.stopping.StoppingCriterion;
import com.github.waikatodatamining.multiway.data.MathUtils;
import com.github.waikatodatamining.multiway.exceptions.InvalidInputException;
import com.google.common.collect.ImmutableSet;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Implementation of the PARAFAC algorithm according to
 * <a href='https://ac.els-cdn.com/S0169743997000324/1-s2.0-S0169743997000324-main.pdf?_tid=529932ca-e37c-11e7-b6f2-00000aab0f26&acdnat=1513551056_3f7d79d6c8151a442526fdfc89e28368'/>
 * <p>
 * Algorithm:
 * <ol>
 * <li>Initialize matrix B, C</li>
 * <li>Estimate A from X, B and C by least squares regression</li>
 * <li>Estimate B likewise</li>
 * <li>Estimate C likewise</li>
 * <li>Repeat until convergence</li>
 * </ol>
 * <p>
 *
 * @author Steven Lang
 */

@Slf4j
public class PARAFAC extends AbstractAlgorithm {

  /** Number of components to reduce to */
  private int numComponents;

  /** Number of random starts */
  private int numStarts;

  /** Number of rows */
  private int numRows;

  /** Number of columns */
  private int numColumns;

  /** Number of dimensions */
  private int numDimensions;

  /** Array of size I x J x K */
  private INDArray X;

  /** Cached matricized X for each axis */
  private INDArray[] Xmatricized;

  /** Array of size I x F */
  private INDArray A;

  /** Array of size J x F */
  private INDArray B;

  /** Array of size K x F */
  private INDArray C;

  /** Cache loading matrices with the lowest loss across restarts */
  private double[][][] bestLoadingMatrices;

  /** Loss history */
  private List<List<Double>> lossHist;

  /** Current loss */
  private double loss;

  /** Best loss */
  private double bestLoss;

  /** Component initialization method */
  private Initialization initMethod;


  /**
   * Constructor setting necessary options.
   *
   * @param numComponents Number of components
   * @param numStarts     Number of starts
   * @param initMethod    Component initialization method
   * @param maxIter       Maximum number of iterations
   */
  public PARAFAC(int numComponents, int numStarts, Initialization initMethod, int maxIter) {
    super();
    this.numComponents = numComponents;
    this.numStarts = numStarts;
    this.lossHist = new ArrayList<>();
    this.initMethod = initMethod;
    this.bestLoss = Double.MAX_VALUE;

    // Validate
    if (numStarts > 1 && initMethod == Initialization.SVD) {
      this.numStarts = 1;
      log.warn("Parameter <numStarts> has no effect if initialization is {}",
	Initialization.SVD);
    }

    addStoppingCriterion(new IterationStoppingCriterion(maxIter));
  }

  /**
   * Constructor setting necessary options. Defaults to 1000 iterations.
   *
   * @param numComponents Number of components
   * @param numStarts     Number of starts
   * @param initMethod    Component initialization method
   */
  public PARAFAC(int numComponents, int numStarts, Initialization initMethod) {
    this(numComponents, numStarts, initMethod, 1000);
  }

  /**
   * Constructor setting necessary options. Defaults to 1000 iterations and a
   * single algorithm run.
   *
   * @param numComponents Number of components
   * @param initMethod    Component initialization method
   */
  public PARAFAC(int numComponents, Initialization initMethod) {
    this(numComponents, 1, initMethod);
  }

  /**
   * Constructor setting necessary options. Defaults to 1000 iterations, a
   * single algorithm run and SVD initialization.
   *
   * @param numComponents Number of components
   */
  public PARAFAC(int numComponents) {
    this(numComponents, Initialization.SVD);
  }

  /**
   * Input is assumed to be of the following shape: (I x J x K), where I is the
   * number of rows, J is the number of columns and K is the number of
   * measurements/components/slices
   *
   * @param input Input data
   */
  public void buildModel(double[][][] input) {
    validateInput(input);

    numRows = input.length;
    numColumns = input[0].length;
    numDimensions = input[0][0].length;

    X = MathUtils.from3dDoubleArray(input);

    // Build matricized cache
    Xmatricized = new INDArray[]{
      MathUtils.matricize(X, 0),
      MathUtils.matricize(X, 1),
      MathUtils.matricize(X, 2)
    };

    for (int i = 0; i < numStarts; i++) {

      // Initialize components
      if (initMethod == Initialization.RANDOM) {
	initComponentsRandom(i);
      }
      else {
	initComponentsSVD();
      }

      // Collect loss for this run
      List<Double> losses = new ArrayList<>();

      while (!stoppingCriteriaMatch()) {
	// Run the nextIteration estimation iteration
	nextIteration();

	// Update algorithm state
	update();

	// Keep track of loss in this run
	losses.add(loss);
      }
      lossHist.add(losses);

      // Update loading matrices if this run was better
      if (loss < bestLoss) {
	bestLoss = loss;
	bestLoadingMatrices = new double[][][]{
	  MathUtils.to2dDoubleArray(A),
	  MathUtils.to2dDoubleArray(B),
	  MathUtils.to2dDoubleArray(C)
	};
      }

      resetStoppingCriteria();
    }
  }

  /**
   * Initialize the component matrices with a random N(0,1) distribution
   *
   * @param seed Seed for the RNG
   */
  private void initComponentsRandom(int seed) {
    A = Nd4j.create(numRows, numComponents);
    B = Nd4j.randn(numColumns, numComponents, seed);
    C = Nd4j.randn(numDimensions, numComponents, seed + 1000);
  }

  /**
   * Initialize all components from eigenvectors using SVD.
   */
  private void initComponentsSVD() {
    A = initComponentSVDop(0);
    B = initComponentSVDop(1);
    C = initComponentSVDop(2);
  }

  /**
   * Initialize a certain component from eigenvectors using SVD.
   * Code is oriented towards <a href="https://github.com/tensorlib/tensorlib/blob/master/tensorlib/decomposition/decomposition.py"/>
   *
   * @param axis Axis to unfold the input data
   * @return Component initialized with eigenvectors
   */
  private INDArray initComponentSVDop(int axis) {
    // Todo: validate nComp and axis
    final INDArray unfolded = Xmatricized[axis];
    final INDArray XXT = unfolded.mmul(unfolded.transpose());
    final RealMatrix rm = CheckUtil.convertToApacheMatrix(XXT);
    EigenDecomposition ed = new EigenDecomposition(rm);
    final RealMatrix eigVecColMat = ed.getV();
    INDArray selectedEigVecs = Nd4j.create(eigVecColMat.getRowDimension(), numComponents);
    int vectorCount = 0;
    for (int i = 0; i < numComponents; i++) {
      selectedEigVecs.putColumn(vectorCount, Nd4j.create(eigVecColMat.getColumn(i)));
      vectorCount++;
    }
    final INDArray argmax = Nd4j.argMax(Transforms.abs(selectedEigVecs), 0);
    INDArray vals = Nd4j.create(numComponents);
    for (int i = 0; i < numComponents; i++) {
      final int j = argmax.getInt(i);
      final double val = selectedEigVecs.getDouble(j, i);
      vals.putScalar(i, val);
    }
    final INDArray sign = Transforms.sign(vals);
    return selectedEigVecs.mulRowVector(sign);
  }

  /**
   * Execute the next iteration:
   * <p>
   * A = X*((C(+)B)^-1)^T
   * B = X*((C(+)A)^-1)^T
   * C = X*((B(+)A)^-1)^T
   * <p>
   * where (+) is the columnwise KhatriRao product.
   */
  private void nextIteration() {
    estimate(A, C, B, 0);
    estimate(B, C, A, 1);
    estimate(C, B, A, 2);
  }

  /**
   * Execute an estimation step for a specific component
   *
   * @param arrToUpdate The component which will be updated in this step
   * @param arr1        Left argument for kr-product
   * @param arr2        Right argument for kr-product
   * @param unfoldAxis  Indicate the axis at which the X tensor will be unfolded
   */
  private void estimate(INDArray arrToUpdate, INDArray arr1, INDArray arr2, int unfoldAxis) {
    // Build Khatri-Rao product
    INDArray res = MathUtils.khatriRaoProductColumnWise(arr1, arr2);
    // Invert and transpose
    res = MathUtils.pseudoInvert(res, true).transposei();
    // Final matrix multiplication
    Xmatricized[unfoldAxis].mmul(res, arrToUpdate);
  }

  /**
   * Get the decomposed loading matrices with the lowest loss across restarts.
   *
   * @return Loading matrices
   */
  public double[][][] getLoadingMatrices() {
    if (bestLoadingMatrices == null) {
      log.warn("Loading matrices are accessed before the model was built.");
    }
    return bestLoadingMatrices;
  }

  /**
   * Calculate the reconstruction calculateLoss
   *
   * @return Reconstruction calculateLoss
   */
  private double calculateLoss() {
    return Xmatricized[0].squaredDistance(reconstruct());
  }

  /**
   * Get the loss history over all reruns and iterations.
   * First index indicates the run and second index indicates the iteration.
   *
   * @return Loss history
   */
  public List<List<Double>> getLossHistory() {
    return lossHist;
  }

  /**
   * Reconstruct the matrix from the estimated components
   *
   * @return Reconstruction from the estimated components
   */
  private INDArray reconstruct() {
    return A.mmul(MathUtils.khatriRaoProductColumnWise(C, B).transpose());
  }

  /**
   * Validate the input data
   *
   * @param inputMatrix Input data
   */
  private void validateInput(double[][][] inputMatrix) {
    if (inputMatrix.length == 0
      || inputMatrix[0].length == 0
      || inputMatrix[0][0].length == 0) {
      throw new InvalidInputException("Input matrix dimensions must be " +
	"greater than 0.");
    }
  }


  @Override
  protected void update() {
    // Update loss
    loss = calculateLoss();

    // Update stopping criteria states
    for (StoppingCriterion sc : stoppingCriteria) {
      switch (sc.getType()) {
	case ITERATION:
	  sc.update();
	  break;
	case TIME:
	  sc.update();
	  break;
	case IMPROVEMENT:
	  ((ImprovementStoppingCriterion) sc).update(loss);
	  break;
      }
    }
  }

  @Override
  protected Set<CriterionType> getAvailableStoppingCriteria() {
    return ImmutableSet.of(
      CriterionType.ITERATION,
      CriterionType.TIME,
      CriterionType.IMPROVEMENT
    );
  }

  /**
   * Enum to define the initialization method of the components.
   */
  public enum Initialization {
    /**
     * Random initialization from N(0,1).
     */
    RANDOM,
    /**
     * Use eigenvalues calculated with SVD.
     */
    SVD
  }
}
