package nz.ac.waikato.cms.adams.multiway.algorithm;

import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.Filter;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.LoadingMatrixAccessor;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.SupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.ImprovementCriterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.IterationCriterion;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.data.tensor.TensorFactory;
import nz.ac.waikato.cms.adams.multiway.exceptions.ModelBuildException;
import nz.ac.waikato.cms.adams.multiway.exceptions.ModelNotBuiltException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType.IMPROVEMENT;
import static nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType.ITERATION;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.center;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.concat;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.invert;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.invertVectorize;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.matricize;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.outer;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.t;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * Multilinear Partial Least Squares Regression.
 * <p>
 * Implementation according to <a href='http://onlinelibrary.wiley.com/doi/10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C/epdf'>R. Bro Multiway Calibration, Multilinear PLS</a>
 * Reference R implementation: <a href='http://models.life.ku.dk/sites/default/files/NPLS_Rver.zip>Download</a>
 * <p>
 *
 * @author Steven Lang
 */
public class MultiLinearPLS extends SupervisedAlgorithm implements Filter, LoadingMatrixAccessor {

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(MultiLinearPLS.class);


  /** Serial version UID */
  private static final long serialVersionUID = 6121171390172636096L;

  /** Number of components / PLS iterations */
  protected int numComponents;

  /** (I x F) Matrix of scores/loadings of first of Y */
  protected Tensor U;

  /** (I x F) Matrix of scores/loadings of first order of X */
  protected Tensor T;

  /** Weights (?) */
  protected Tensor W;

  /** (J x F) Loadings in second order of X */
  protected Tensor Wj;

  /** (K x F) Loadings in third order of X */
  protected Tensor Wk;

  /** (Jy x F) Loadings in second order of Y */
  protected Tensor Q;

  /** (F x F) Matrix of regression coefficients */
  protected Tensor B;

  /** F: num targets (second dim of Y) */
  protected int numTargets;

  /** Mean vector of the target matrix */
  protected Tensor yMean;

  /** Std vector of the target matrix */
  protected Tensor yStd;

  /** Mean vector of the input matrix */
  protected Tensor xMean;

  /** Std vector of the input matrix */
  protected Tensor xStd;

  /** Whether to standardize Y or not */
  protected boolean standardizeY;

  @Override
  protected void initialize() {
    super.initialize();
    this.numComponents = 10;
    this.standardizeY = true;
    addStoppingCriterion(CriterionUtils.iterations(250));
    addStoppingCriterion(CriterionUtils.improvement(10E-8));
  }


  @Override
  protected String doBuild(Tensor xTensor, Tensor yTensor) {
    final int xI = xTensor.size(0);
    final int xJ = xTensor.size(1);
    final int xK = xTensor.size(2);
    numTargets = yTensor.size(1);

    // Unfold X in first mode IxJxK -> IxJ*K
    Tensor Xa = matricize(xTensor, 0);
    Tensor Xres = Xa.dup();
    Tensor Xmodel = TensorFactory.zeros(xI, xJ * xK);
    Tensor Y = yTensor;
    Tensor Yres = Y.dup();

    // Get stats
    yMean = Y.mean(0);
    yStd = Y.std(0);
    xMean = Xa.mean(0);
    xStd = Xa.std(0);

    // Center X,Y across the first mode
    Xa = center(Xa, 0);
    Y = center(Y, 0);
    if (standardizeY) {
      Y = Y.divRowVector(yStd);
    }

    T = null;
    B = TensorFactory.zeros(numComponents, numComponents);
    Tensor ba;

    // Use first column of Y as u
    Tensor u = null;
    Tensor ta = null;
    Tensor wa = null;
    Tensor q = null;
    final ImprovementCriterion imprCrit = (ImprovementCriterion) stoppingCriteria.get(IMPROVEMENT);
    final IterationCriterion iterCrit = (IterationCriterion) stoppingCriteria.get(ITERATION);

    TwoWayPCA pca = new TwoWayPCA();
    pca.setNumComponents(1);

    // Generate N components
    for (int a = 0; a < numComponents; a++) {
      pca.build(Yres);
      u = pca.getLoadingMatrices().get("T");

      // Inner NIPALS loop
      while (!(iterCrit.matches() || imprCrit.matches())) {
	wa = getW(Xa, u, xJ, xK);
	ta = Xres.mmul(wa);
	q = (t(Yres).mmul(ta)).normalize();
	u = Yres.mmul(q);

	// Update iteration criterion
	iterCrit.update();
      }
      pca.resetState();

      // Reset iterations for next NIPALS loop
      iterCrit.reset();

      // Update improvement criterion
      imprCrit.update(u.norm2().getDouble(0));

      // Check if improvement criterion stopped the NIPALS loop before creating
      // first components
      if (wa == null || ta == null || q == null) {
	throw new ModelBuildException(
	  String.format("Could not initialize the first components. " +
	      "ImprovementCriterion tolerance of %f might be set too high.",
	    imprCrit.getTol()));
      }

      // Collect loading/score components
      final Tensor[] wjWk = getWjWk(Xa, u, xJ);
      Wj = Wj.concat(wjWk[0], 1);
      Wk = Wk.concat(wjWk[1], 1);
      W = W.concat(wa, 1);
      Q = Q.concat( q, 1);
      U = U.concat(u, 1);
      T = T.concat(ta, 1);

      // Deflate X
      Xmodel = Xmodel.add(ta.mmul(t(wa)));
      Xres = Xa.sub(Xmodel);


      // Estimate ba
      ba = invert(t(T).mmul(T)).mmul(t(T)).mmul(u);
      B.put(new INDArrayIndex[]{interval(0, a + 1), point(a)}, ba);

      // Deflate Y
      Tensor ypred = T.mmul(B.get(interval(0, a + 1), interval(0, a + 1)).dup()).mmul(t(Q));
      Yres = Y.sub(ypred);
    }
    return null;
  }

  /**
   * Compute w as
   * w = kronecker(w^J,w^K) from3dDoubleArray(data);
   * with
   * (w^J,w^K) = SVD(Z)
   * and Vec(Z) = X^T*y
   *
   * @param X             Matricized input
   * @param y             Dependent variables
   * @param numColumns    Number of columns
   * @param numDimensions Number of dimensions
   * @return w
   */
  protected Tensor getW(Tensor X, Tensor y, int numColumns, int numDimensions) {
    final Tensor[] wjwk = getWjWk(X, y, numColumns);
    final Tensor kronecker = outer(wjwk[1], wjwk[0]);
    return kronecker.reshape(numColumns * numDimensions, -1);
  }


  /**
   * Compute  (w^J,w^K) = SVD(Z) with Vec(Z) = X^T*y
   * w^J: first left singular vector of SVD(Z) (first column vector of U)
   * w^K: first right singular vector of SVD(Z) (first column vector of V)
   *
   * @param X          Matricized input
   * @param y          Dependent variables
   * @param numColumns Number of columns
   * @return w^J, w^K
   */
  protected Tensor[] getWjWk(Tensor X, Tensor y, int numColumns) {
    final Tensor vecZ = t(X).mmul(y);

    INDArray Z = invertVectorize(vecZ, numColumns);

    // Solve SVD
    final Map<String, Tensor> svd = MathUtils.svd(Z);
    final Tensor U = svd.get("U");
    final Tensor V = svd.get("V");

    // w^J and w^K are the first left and the first right singular vectors
    Tensor wJ = U.getColumn(0).dup();
    Tensor wK = V.getColumn(0).dup();

    // normalize w^J and w^K
    wJ = wJ.normalize();
    wK = wJ.normalize();

    return new Tensor[]{wJ, wK};
  }


  @Override
  protected Set<CriterionType> getAvailableStoppingCriteria() {
    return ImmutableSet.of(ITERATION, IMPROVEMENT);
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
  protected String check(Tensor x, Tensor y) {
    String superCheck = super.check(x, y);
    if (superCheck != null){
      return superCheck;
    }
    if (x.size(0) == 0
      || x.size(1) == 0
      || x.size(2) == 0) {
      return "Input matrix dimensions must be " +
	"greater than 0.";
    }

    if (x.size(0) != y.size(0)) {
      return "Independent and dependent variables must" +
	" be of the same length.";
    }

    return null;
  }


  @Override
  public Tensor predict(Tensor input) {

    // Check if the model has been built yet
    if (!isFinished()){
      throw new ModelNotBuiltException(
        "Trying to invoke predict(Tensor input) while the model has not been " +
          "built yet."
      );
    }

    final Tensor Tnew = filter(input);

    Tensor Ypred = Tnew.mmul(B).mmul(t(Q));

    // Rescale
    if (standardizeY) {
      Ypred = Ypred.mulRowVector(yStd);
    }

    // Add mean
    Ypred = Ypred.addRowVector(yMean);

    return Ypred;
  }

  @Override
  public Tensor filter(Tensor input) {

    // Check if the model has been built yet
    if (!isFinished()){
      throw new ModelNotBuiltException(
        "Trying to invoke filter(Tensor input) while the model has not been " +
          "built yet."
      );
    }

    Tensor X = matricize(input, 0);
    X = X.subRowVector(xMean);
    Tensor Xres = X.dup();
    Tensor T = null;

    for (int a = 0; a < numComponents; a++) {
      final Tensor wja = this.Wj.getColumn(a).dup();
      final Tensor wka = this.Wk.getColumn(a).dup();
      final Tensor load = outer(wka, wja).reshape(-1, X.size(1));
      final Tensor ta = Xres.mmul(t(load));
      T = T.concat( ta, 1);
      Xres = Xres.sub(ta.mmul(load));
    }

    return T;
  }

  @Override
  protected void resetState() {
    super.resetState();
    U = null;
    T = null;
    W = null;
    Wj = null;
    Wk = null;
    Q = null;
    B = null;
    yMean = null;
    yStd = null;
    xMean = null;
    xStd = null;
  }

  @Override
  public Map<String, Tensor> getLoadingMatrices() {
    Map<String, Tensor> m = new HashMap<>();
    m.put("U", U);
    m.put("T", T);
    m.put("W", W);
    m.put("Wj", Wj);
    m.put("Wk", Wk);
    m.put("Q", Q);
    m.put("B", B);
    return m;
  }
}

