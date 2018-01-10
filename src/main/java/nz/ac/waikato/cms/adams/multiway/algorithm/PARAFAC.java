package nz.ac.waikato.cms.adams.multiway.algorithm;

import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.Criterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.ImprovementCriterion;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.exceptions.InvalidInputException;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Implementation of the PARAFAC algorithm according to
 * <a href='https://ac.els-cdn.com/S0169743997000324/1-s2.0-S0169743997000324-main.pdf?_tid=529932ca-e37c-11e7-b6f2-00000aab0f26&acdnat=1513551056_3f7d79d6c8151a442526fdfc89e28368'>Rasmus Bro, PARAFAC. Tutorial and applications</a>
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

public class PARAFAC extends AbstractAlgorithm {

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(PARAFAC.class);

  /** Serial version UID */
  private static final long serialVersionUID = -6263360332015252931L;

  /** Number of components to reduce to */
  protected int numComponents;

  /** Number of random starts */
  protected int numStarts;

  /** Cached matricized X for each axis */
  protected INDArray[] Xmatricized;

  /** Array of shape I x F */
  protected INDArray A;

  /** Array of shape J x F */
  protected INDArray B;

  /** Array of shape K x F */
  protected INDArray C;

  /** Cache loading matrices with the lowest loss across restarts */
  protected double[][][] bestLoadingMatrices;

  /** Loss history */
  protected List<List<Double>> lossHistory;

  /** Current loss */
  protected double loss;

  /** Best loss */
  protected double bestLoss;

  /** Component initialization method */
  protected Initialization initMethod;

  @Override
  public void initialize() {
    super.initialize();
    this.lossHistory = new ArrayList<>();
    this.bestLoss = Double.MAX_VALUE;
    this.initMethod = Initialization.SVD;
    this.numStarts = 1;
    this.numComponents = 3;
    addStoppingCriterion(CriterionUtils.iterations(1000));
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
    if (numStarts > 1 && initMethod == Initialization.SVD) {
      this.numStarts = 1;
      log.warn("Parameter <numStarts> has no effect if initialization is {}." +
	  " <numStarts> has therefore been reset to 1.",
	Initialization.SVD);
    }

    final int numRows = input.length;
    final int numColumns = input[0].length;
    final int numDimensions = input[0][0].length;

    // Array of shape I x J x K
    INDArray X = MathUtils.from3dDoubleArray(input);

    // Build matricized cache
    Xmatricized = new INDArray[]{
      MathUtils.matricize(X, 0),
      MathUtils.matricize(X, 1),
      MathUtils.matricize(X, 2)
    };

    for (int i = 0; i < numStarts; i++) {

      // Initialize components
      if (initMethod == Initialization.RANDOM) {
	initComponentsRandom(numRows, numColumns, numDimensions, i);
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
      lossHistory.add(losses);

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
   * @param numRows       Number of rows (first mode)
   * @param numColumns    Number of columns (second mode)
   * @param numDimensions Number of dimensions (third mode)
   * @param seed          Seed for the RNG
   */
  protected void initComponentsRandom(int numRows, int numColumns, int numDimensions, int seed) {
    A = Nd4j.create(numRows, numComponents);
    B = Nd4j.randn(numColumns, numComponents, seed);
    C = Nd4j.randn(numDimensions, numComponents, seed + 1000);
  }

  /**
   * Initialize all components from eigenvectors using SVD.
   */
  protected void initComponentsSVD() {
    A = initComponentSVDop(0);
    B = initComponentSVDop(1);
    C = initComponentSVDop(2);
  }

  /**
   * Initialize a certain component from eigenvectors using SVD.
   * Code is oriented towards <a href="https://github.com/tensorlib/tensorlib/blob/master/tensorlib/decomposition/decomposition.py">tensorlib</a>
   *
   * @param axis Axis to unfold the input data
   * @return Component initialized with eigenvectors
   */
  protected INDArray initComponentSVDop(int axis) {
    // Todo: validate nComp and axis
    final INDArray unfolded = Xmatricized[axis];
    final INDArray XXT = unfolded.mmul(MathUtils.t(unfolded));
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
  protected void nextIteration() {
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
  protected void estimate(INDArray arrToUpdate, INDArray arr1, INDArray arr2, int unfoldAxis) {
    // Build Khatri-Rao product
    INDArray res = MathUtils.khatriRaoProductColumnWise(arr1, arr2);
    // Invert and transpose
    res = MathUtils.pseudoInvert(res, true).transposei();
    // Final matrix multiplication
    Xmatricized[unfoldAxis].mmul(res, arrToUpdate);
  }

  /**
   * Calculate the reconstruction calculateLoss
   *
   * @return Reconstruction calculateLoss
   */
  protected double calculateLoss() {
    return Xmatricized[0].squaredDistance(reconstruct());
  }

  /**
   * Get the loss history over all reruns and iterations.
   * First index indicates the run and second index indicates the iteration.
   *
   * @return Loss history
   */
  public List<List<Double>> getLossHistory() {
    return lossHistory;
  }

  /**
   * Reconstruct the matrix from the estimated components
   *
   * @return Reconstruction from the estimated components
   */
  protected INDArray reconstruct() {
    return A.mmul(MathUtils.t(MathUtils.khatriRaoProductColumnWise(C, B)));
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

    for (int i = 0; i < inputMatrix.length; i++) {
      for (int j = 0; j < inputMatrix[0].length; j++) {
	for (int k = 0; k < inputMatrix[0][0].length; k++) {
	  if (Double.isNaN(inputMatrix[i][j][k])) {
	    throw new InvalidInputException("Input has missing data " +
	      "(NaNs found). PARAFAC currently does not support missing data.");
	  }
	}
      }
    }
  }

  /**
   * Update the internal state.
   */
  protected void update() {
    // Update loss
    loss = calculateLoss();

    // Update stopping criteria states
    for (Criterion sc : stoppingCriteria) {
      switch (sc.getType()) {
	case IMPROVEMENT:
	  ((ImprovementCriterion) sc).update(loss);
	  break;
	default:
	  sc.update();
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
   * Get number of components.
   *
   * @return Number of components
   */
  public int getNumComponents() {
    return numComponents;
  }

  /**
   * Set number of components.
   *
   * @param numComponents Number of components
   */
  public void setNumComponents(int numComponents) {
    if (numComponents < 1) {
      log.warn("Number of components must be greater " +
	"than zero.");
    }
    else {
      this.numComponents = numComponents;
    }
  }

  /**
   * Get number of restarts.
   *
   * @return Number of restarts
   */
  public int getNumStarts() {
    return numStarts;
  }

  /**
   * Set number of restarts.
   *
   * @param numStarts Number of restarts
   */
  public void setNumStarts(int numStarts) {
    if (numStarts < 1) {
      log.warn("Number of starts must be greater " +
	"than zero.");
    }
    else {
      this.numStarts = numStarts;
    }
  }

  /**
   * Get the loading matrices A,B,C with the lowest reconstruction error.
   *
   * @return Loading matrices with the lowest reconstruction error
   */
  public double[][][] getBestLoadingMatrices() {
    if (bestLoadingMatrices == null) {
      log.warn("Loading matrices are accessed before the model was built.");
    }
    return bestLoadingMatrices;
  }

  /**
   * Get the loading matrix initialization method.
   *
   * @return Loading matrix initialization method
   */
  public Initialization getInitMethod() {
    return initMethod;
  }


  /**
   * Set the loading matrix initialization method.
   *
   * @param initMethod Loading matrix initialization method
   */
  public void setInitMethod(Initialization initMethod) {
    this.initMethod = initMethod;
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
