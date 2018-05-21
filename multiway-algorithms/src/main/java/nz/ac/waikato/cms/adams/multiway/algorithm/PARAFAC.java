package nz.ac.waikato.cms.adams.multiway.algorithm;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.Filter;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.LoadingMatrixAccessor;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.UnsupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.Criterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.ImprovementCriterion;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.data.tensor.TensorFactory;
import nz.ac.waikato.cms.adams.multiway.exceptions.InvalidInputException;
import nz.ac.waikato.cms.adams.multiway.exceptions.ModelNotBuiltException;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.Tensor;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
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

public class PARAFAC extends UnsupervisedAlgorithm implements LoadingMatrixAccessor, Filter {

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(PARAFAC.class);

  /** Serial version UID */
  private static final long serialVersionUID = -6263360332015252931L;

  /** Number of components to reduce to */
  protected int numComponents;

  /** Number of random starts */
  protected int numStarts;

  /** Cached matricized X for each axis */
  protected Tensor[] Xmatricized;

  /** Array of shape I x F. Score matrix. */
  protected Tensor A;

  /** Array of shape J x F. First loading matrix. */
  protected Tensor B;

  /** Array of shape K x F. Second loading matrix. */
  protected Tensor C;

  /** Cache loading matrices with the lowest loss across restarts */
  protected Tensor[] bestLoadingMatrices;

  /** Loss history */
  protected List<List<Double>> lossHistory;

  /** Current loss */
  protected double loss;

  /** Best loss */
  protected double bestLoss;

  /** Component initialization method */
  protected Initialization initMethod;

  @Override
  protected void initialize() {
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
   * @param x Input data
   */
  @Override
  protected String doBuild(Tensor x) {
    // Array of shape I x J x K
    Tensor X = x;
    if (numStarts > 1 && initMethod == Initialization.SVD) {
      this.numStarts = 1;
      log.warn("Parameter <numStarts> has no effect if initialization is {}." +
	  " <numStarts> has therefore been reset to 1.",
	Initialization.SVD);
    }

    // Get dimensions
    final int numRows = X.size(0);
    final int numColumns = X.size(1);
    final int numDimensions = X.size(2);


    // Build matricized cache
    Xmatricized = new Tensor[]{
      MathUtils.matricize(X, 0),
      MathUtils.matricize(X, 1),
      MathUtils.matricize(X, 2)
    };


    // Repeat #numStarts times
    for (int i = 0; i < numStarts; i++) {
      initComponents(numRows, numColumns, numDimensions, i);

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
	bestLoadingMatrices = new Tensor[]{A, B, C};
      }

      resetStoppingCriteria();
    }

    /*
     * Postprocess: Put all variance in first mode according to
     * https://github.com/andrewssobral/nway/blob/a6dc5b3970ef395c03fc7e6dc1ea2fc105185b86/parafac.m#L941
     */
    for (Tensor loading : new Tensor[] {B, C}) {
      for (int f = 0; f < numComponents; f++) {
        double norm = loading.getColumn(f).norm2().getDouble(0);
        Tensor normedA = A.getColumn(f).mul(norm);
        A.putColumn(f, normedA);
        Tensor normedColumn = loading.getColumn(f).div(norm);
        loading.putColumn(f, normedColumn);
      }
    }

    /*
     * Sort components in order after variance described (as in PCA)
     * See also: https://github.com/andrewssobral/nway/blob/a6dc5b3970ef395c03fc7e6dc1ea2fc105185b86/parafac.m#L1193
     */
    final Tensor diag = Nd4j.diag(A.t().mmul(A));
    final Tensor orderArray = Nd4j.sortWithIndices(diag, 0, false)[0];
    int[] order = new int[orderArray.size(0)];
    for (int i = 0; i < orderArray.size(0); i++) {
      order[i] = orderArray.getInt(i);
    }

    A.assign(Nd4j.pullRows(A, 0, order));
    B.assign(Nd4j.pullRows(B, 0, order));
    C.assign(Nd4j.pullRows(C, 0, order));

    /*
     * Apply sign convention:
     * See also: https://github.com/andrewssobral/nway/blob/a6dc5b3970ef395c03fc7e6dc1ea2fc105185b86/parafac.m#L1231
     */
    Tensor signs = TensorFactory.ones(1, numComponents);
    for (Tensor loading : new Tensor[]{C, B}) {
      Tensor signs2 = TensorFactory.ones(1, numComponents);
      for (int f = 0; f < numComponents; f++) {
        final Tensor colF = loading.getColumn(f);
        final Tensor colfFabs = Transforms.abs(colF);
        final int argmax = colfFabs.argMax(0).getInt(0);
        signs.putScalar(f, signs.getDouble(f) * Math.signum(loading.getDouble(argmax, f)));
        signs2.putScalar(f, Math.signum(loading.getDouble(argmax, f)));
      }
      loading.assign(loading.mmul(Nd4j.diag(signs2)));
    }
    A.assign(A.mmul(Nd4j.diag(signs)));

    return null;
  }

  /**
   * Initialize the components based on the chosen initialization method,
   *
   * @param numRows       Number of rows
   * @param numColumns    Number of columns
   * @param numDimensions Number of dimensions
   * @param seed          Seed
   */
  protected void initComponents(int numRows, int numColumns, int numDimensions, int seed) {
    // Initialize components
    switch (initMethod) {
      case RANDOM:
	initComponentsRandom(numRows, numColumns, numDimensions, seed);
	break;
      case RANDOM_ORTHOGONALIZED:
	initComponentsRandomOrthogonalized(numRows, numColumns, numDimensions, seed);
	break;
      case SVD:
	initComponentsSVD();
	break;
      default:
	throw new InvalidInputException("Initialization method " +
          initMethod + " is not yet implemented.");
    }
  }

  @Override
  protected String check(Tensor input) {
    String superCheck = super.check(input);
    if (superCheck != null){
      return superCheck;
    }

    // Check for wrong dimensions
    if (input.size(0) == 0
      || input.size(1) == 0
      || input.size(2) == 0) {
      return "Input tensor dimensions must be greater than 0.";
    }

    // Check for NaNs
    for (int i = 0; i < input.size(0); i++) {
      for (int j = 0; j < input.size(1); j++) {
	for (int k = 0; k < input.size(2); k++) {
	  if (Double.isNaN(input.getDouble(i, j, k))) {
	    return "Input has missing data " +
	      "(NaNs found). PARAFAC currently does not support missing data.";
	  }
	}
      }
    }
    return null;
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
    A = TensorFactory.zeros(numRows, numComponents);
    B = TensorFactory.randn(numColumns, numComponents, seed);
    C = TensorFactory.randn(numDimensions, numComponents, seed + 1000);
  }

  /**
   * Initialize the component matrices with a random orthogonalized matrices
   *
   * @param numRows       Number of rows (first mode)
   * @param numColumns    Number of columns (second mode)
   * @param numDimensions Number of dimensions (third mode)
   * @param seed          Seed for the RNG
   */
  protected void initComponentsRandomOrthogonalized(int numRows, int numColumns, int numDimensions, int seed) {
    A = TensorFactory.zeros(numRows, numComponents);
    B = MathUtils.orth(TensorFactory.randn(numColumns, numComponents, seed), false);
    C = MathUtils.orth(TensorFactory.randn(numDimensions, numComponents, seed + 1000), false);
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
  protected Tensor initComponentSVDop(int axis) {
    // Todo: validate nComp and axis
    final Tensor unfolded = Xmatricized[axis];
    final Tensor XXT = unfolded.mmul(MathUtils.t(unfolded));
    final RealMatrix rm = CheckUtil.convertToApacheMatrix(XXT);
    EigenDecomposition ed = new EigenDecomposition(rm);
    final RealMatrix eigVecColMat = ed.getV();
    Tensor selectedEigVecs = Nd4j.create(eigVecColMat.getRowDimension(), numComponents);
    int vectorCount = 0;
    for (int i = 0; i < ed.getRealEigenvalues().length; i++) {
      // Skip eigenvalues of zero (apache commons puts those first even though the order should be descending)
      if (ed.getRealEigenvalue(i) < 10e-7){
        continue;
      }

      final Tensor eigVecI = Nd4j.create(eigVecColMat.getColumn(i));
      selectedEigVecs.putColumn(vectorCount, eigVecI);
      vectorCount++;

      // Check if all components have been fetched
      if (vectorCount == numComponents){
        break;
      }
    }
    final Tensor argmax = Nd4j.argMax(Transforms.abs(selectedEigVecs), 0);
    Tensor vals = Nd4j.create(numComponents);
    for (int i = 0; i < numComponents; i++) {
      final int j = argmax.getInt(i);
      final double val = selectedEigVecs.getDouble(j, i);
      vals.putScalar(i, val);
    }
    final Tensor sign = Transforms.sign(vals);
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
    estimate(A, Xmatricized[0], C, B);
    estimate(B, Xmatricized[1], C, A);
    estimate(C, Xmatricized[2], B, A);
  }

  /**
   * Execute an estimation step for a specific component
   *
   * @param arrToUpdate The component which will be updated in this step
   * @param Xunfolded Unfolded input matrix
   * @param arr1        Left argument for kr-product
   * @param arr2        Right argument for kr-product
   */
  protected void estimate(Tensor arrToUpdate, Tensor Xunfolded, Tensor arr1, Tensor arr2) {
    // Build Khatri-Rao product
    Tensor res = MathUtils.khatriRaoProductColumnWise(arr1, arr2);
    // Invert and transpose
    res = MathUtils.pseudoInvert(res, true).transposei();
    // Final matrix multiplication
    Xunfolded.mmul(res, arrToUpdate);
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
  protected Tensor reconstruct() {
    return A.mmul(MathUtils.t(MathUtils.khatriRaoProductColumnWise(C, B)));
  }

  /**
   * Update the internal state.
   */
  protected void update() {
    // Update loss
    loss = calculateLoss();

    // Update stopping criteria states
    for (Criterion sc : stoppingCriteria.values()) {
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
    Set<CriterionType> criteria = new HashSet<>();
    criteria.add(CriterionType.ITERATION);
    criteria.add(CriterionType.TIME);
    criteria.add(CriterionType.IMPROVEMENT);
    return criteria;
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

  @Override
  protected void resetState() {
    super.resetState();
    A = null;
    B = null;
    C = null;
    Xmatricized = new Tensor[3];
    lossHistory = new ArrayList<>();
    bestLoss = Double.MAX_VALUE;
  }

  /**
   * Get the loading matrices A,B,C with the lowest reconstruction error.
   *
   * @return Loading matrices with the lowest reconstruction error
   */
  @Override
  public Map<String, Tensor> getLoadingMatrices() {
    if (bestLoadingMatrices == null) {
      log.warn("Loading matrices are accessed before the model was built.");
    }

    Map<String, Tensor> map = new HashMap<>();
    map.put("A",bestLoadingMatrices[0]);
    map.put("B",bestLoadingMatrices[1]);
    map.put("C",bestLoadingMatrices[2]);

    return map;
  }


  /**
   * Use PARAFAC to generate score (A) of new data based on a previously calibrated
   * model using its loadings (B,C).
   *
   * See also: <a href='https://onlinelibrary.wiley.com/doi/full/10.1002/cem.1037'>Multi‐way prediction in the presence of uncalibrated interferents</a>
   *
   * @param input New input tensor
   */
  @Override
  public Tensor filter(Tensor input) {

    // Check if the model has been built yet
    if (!isFinished()){
      throw new ModelNotBuiltException(
        "Trying to invoke filter(Tensor input) while the model has not been " +
          "built yet."
      );
    }

    Tensor Anew = TensorFactory.zeros(input.size(0), numComponents);
    final Tensor Xmatricized = MathUtils.matricize(input, 0);
    estimate(Anew, Xmatricized, C, B);
    return Anew;
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
     * Random initialization with orthogonalized matrices.
     */
    RANDOM_ORTHOGONALIZED,
    /**
     * Use eigenvalues calculated with SVD.
     */
    SVD
  }
}
