package nz.ac.waikato.cms.adams.multiway.algorithm;

import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.LoadingMatrixAccessor;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.MultiBlockSupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import static nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType.IMPROVEMENT;
import static nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType.ITERATION;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.invert;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.meanSquaredError;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.t;

/**
 * Sequentially Orthogonalized Multilinear Partial Least Squares Regression.
 * <p>
 * Implementation according to <a href='https://www.sciencedirect.com/science/article/abs/pii/S0169743917301545'>Extension
 * of SO-PLS to multi-way arrays: SO-N-PLS</a>
 * <p>
 *
 * @author Steven Lang
 */
public class SONPLS extends MultiBlockSupervisedAlgorithm implements LoadingMatrixAccessor {

  private static final long serialVersionUID = -2676181670927921239L;

  /**
   * Logger instance
   */
  private static final Logger log = LogManager.getLogger(SONPLS.class);

  /** Number of components / PLS iterations */
  protected int[] numComponents;

  /**
   * Flag to find best number of components for each block automatically
   * (greedy)
   */
  protected boolean autoNumComponents;

  /** Whether to standardize Y or not */
  protected boolean standardizeY;

  /** Array of NPLS modles */
  protected MultiLinearPLS[] nplss;

  /**
   * Get number of components.
   *
   * @return Number of components
   */
  public int[] getNumComponents() {
    return numComponents;
  }

  /**
   * Set number of components.
   *
   * @param numComponents Number of components
   */
  public void setNumComponents(int[] numComponents) {

    // Check that all number of components are > 0
    boolean success = true;
    for (int i = 0; i < numComponents.length; i++) {
      if (numComponents[i] < 1) {
	log.warn("Number of components must be greater " +
	  "than zero but was " + numComponents[i] + " for block number " + i +
	  ".");
	success = false;
      }
    }

    // If check succeeded, set number of components
    if (success) {
      this.numComponents = numComponents;
    }

  }

  /**
   * Flag if automatically number of components is determined.
   *
   * @return True if flag is set
   */
  public boolean isAutoNumComponents() {
    return autoNumComponents;
  }

  /**
   * Sets auto num components flag.
   *
   * @param autoNumComponents Flag
   */
  public void setAutoNumComponents(boolean autoNumComponents) {
    this.autoNumComponents = autoNumComponents;
  }

  /**
   * If the target matrix Y will be standardized.
   *
   * @return True f Y will be standardized
   */
  public boolean isStandardizeY() {
    return standardizeY;
  }

  /**
   * Set if the target matrix Y will be standardized.
   *
   * @param standardizeY if Y will be standardized
   */
  public void setStandardizeY(boolean standardizeY) {
    this.standardizeY = standardizeY;
  }

  @Override
  protected void initialize() {
    super.initialize();
    numComponents = new int[0];
    autoNumComponents = true;
  }

  @Override
  protected Set<CriterionType> getAvailableStoppingCriteria() {
    return ImmutableSet.of(ITERATION, IMPROVEMENT);
  }

  @Override
  public Map<String, Tensor> getLoadingMatrices() {
    Map<String, Tensor> res = new HashMap<>();

    // Collect matrices of all models
    for (int i = 0; i < nplss.length; i++) {
      MultiLinearPLS npls = nplss[i];

      // Add matrices of each model annotated with the block index
      for (Entry<String, Tensor> e : npls.getLoadingMatrices().entrySet()) {
	res.put(e.getKey() + "_" + i, e.getValue());
      }
    }
    return res;
  }

  @Override
  protected String doBuild(Tensor[] x, Tensor y) {
    int numBlocks = x.length;
    INDArray Yres = y.getData();
    List<INDArray> Ts = new ArrayList<>();
    INDArray xiOrth;
    nplss = new MultiLinearPLS[numBlocks];



    if (autoNumComponents) {
      numComponents = new int[numBlocks];
    } else if (numComponents.length != numBlocks) {
      String error = "Number of components array does not match number of " +
        "X-blocks. Was " + numComponents.length + " but should be " +
        numBlocks + ".";
      log.error(error);
      return error;
    }

    // Main loop
    for (int i = 0; i < numBlocks && !isForceStop(); i++) {
      // 1) Orthogonalize current block (returns x if (i == 1))
      xiOrth = orth(x, Ts, i);

      // 2) Fit new PLS model on xorth and residuals
      nplss[i] = buildNPLS(xiOrth, Yres, numComponents[i]);

      // 3) Collect scores for orthogonalization of further blocks
      Ts.add(scores(nplss[i]));

      // 4) Compute residuals
      Yres = residuals(nplss[i], xiOrth, Yres);
    }
    return null;
  }

  /**
   * Orthogonalize the i-th X block w.r.t. the ts array.
   *
   * @param x  x blocks
   * @param ts ts array (prev x block scores)
   * @param i  Current SONPLS iteration
   * @return Orthogonalized Xi
   */
  private INDArray orth(Tensor[] x, List<INDArray> ts, int i) {
    if (i == 0) { // Skip for first block
      return x[0].getData();
    }
    else { // Orth. w.r.t. scores (Ts) of previous blocks
      return orth(x[i].getData(), ts);
    }
  }


  /**
   * Orthogonalize Xi w.r.t. Ti. Xi_orth = Xi - Ti * (Ti^T * Ti)^-1 * Ti^T * Xi
   * <p>
   * See also: Path modelling by sequential PLS regression, J. Chemometrics
   * 2011; 25: 28–40
   *
   * @param Xi i-th X-block
   * @param Ts Alls score of previous Xi blocks
   * @return Orthogonlaized Xi-block w.r.t. previous Xi-block scores
   */
  private INDArray orth(INDArray Xi, List<INDArray> Ts) {
    INDArray Xiorth = Xi.dup();
    INDArray Ximatricized = null;
    // Check if block is threeway
    boolean blockIsThreeway = Xiorth.rank() == 3;
    if (blockIsThreeway) { // Unfold
      Xiorth = MathUtils.matricize(Xiorth, 0);
      Ximatricized = MathUtils.matricize(Xi, 0);
    }
    else {
      Ximatricized = Xi;
    }

    // Orthogonalize the current block w.r.t. all previous block PLS-scores
    for (INDArray Ti : Ts) {
      // Inplace
      Xiorth.subi(Ti.mmul(invert((t(Ti)).mmul(Ti))).mmul(t(Ti)).mmul(Ximatricized));
    }

    // Invert matricize
    if (blockIsThreeway) {
      int dim1 = (int) Xi.size(1);
      int dim2 = (int) Xi.size(2);
      Xiorth = MathUtils.invertMatricize(Xiorth, 0, dim1, dim2);
    }
    return Xiorth;
  }

  /**
   * Build npls multi linear pls.
   *
   * @param X             the x
   * @param Y             the y
   * @param numComponents the num components
   * @return the multi linear pls
   */
  private MultiLinearPLS buildNPLS(INDArray X, INDArray Y, int numComponents) {


    // Transform two way blocks into pseudo threeway block with the third dim
    // of size 1
    if (X.rank() == 2) {
      long dim1 = X.size(1);
      long dim0 = X.size(0);
      long dim2 = 1;
      X = X.reshape(dim0, dim1, dim2);
    }

    if (autoNumComponents) {
      return findBestNPLS(X, Y);
    }
    else {
      // Build NPLS based on numComponents set in options
      MultiLinearPLS npls = new MultiLinearPLS();
      npls.setStandardizeY(standardizeY);
      npls.setNumComponents(numComponents);
      resetStoppingCriteria();
      npls.setStoppingCriteria(this.getStoppingCriteria());
      npls.build(Tensor.create(X), Tensor.create(Y));
      return npls;
    }
  }

  /**
   * Find the best NPLS model by checking all possbile numComponents values.
   *
   * @param X Xi block
   * @param Y Target
   * @return Best NPLS model (based on MSE)
   */
  private MultiLinearPLS findBestNPLS(INDArray X, INDArray Y) {
    long maxK = X.size(1) * X.size(2);
    int minK = 1;
    MultiLinearPLS bestNPLS = null;
    MultiLinearPLS currentNPLS;
    double bestMSE = Double.POSITIVE_INFINITY;
    INDArray Yhat;

    // For each k build a model and check if it has the lowest MSE
    for (int k = minK; k < maxK; k++) {
      currentNPLS = new MultiLinearPLS();
      currentNPLS.setNumComponents(k);
      currentNPLS.setStandardizeY(standardizeY);
      resetStoppingCriteria();
      currentNPLS.setStoppingCriteria(this.getStoppingCriteria());
      currentNPLS.build(Tensor.create(X), Tensor.create(Y));
      Yhat = currentNPLS.predict(Tensor.create(X)).getData();
      double mse = meanSquaredError(Y, Yhat);

      if (mse < bestMSE) {
	bestMSE = mse;
	bestNPLS = currentNPLS;
      }
    }

    // Return NPLS with lowest mse
    return bestNPLS;
  }

  /**
   * Residuals ind array.
   *
   * @param npls the npls
   * @param X    the x
   * @param Y    the y
   * @return the ind array
   */
  private INDArray residuals(MultiLinearPLS npls, INDArray X, INDArray Y) {
    INDArray yhat = npls.predict(Tensor.create(X)).getData();
    return Y.sub(yhat);
  }

  /**
   * Scores ind array.
   *
   * @param npls the npls
   * @return the ind array
   */
  private INDArray scores(MultiLinearPLS npls) {
    return npls.getLoadingMatrices().get("T").getData();
  }

  @Override
  public Tensor predict(Tensor[] x) {

    // Collect additive predictions of each NPLS model
    INDArray Yhat = null;
    for (int i = 0; i < x.length; i++) {
      INDArray yihat = nplss[i].predict(x[i]).getData();
      if (Yhat == null) {
	Yhat = yihat;
      }
      else {
	Yhat = Yhat.add(yihat);
      }
    }

    return Tensor.create(Yhat);
  }
}
