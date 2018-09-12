package nz.ac.waikato.cms.adams.multiway.algorithm;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.Filter;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.LoadingMatrixAccessor;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.UnsupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.exceptions.ModelNotBuiltException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Map;
import java.util.Set;

import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.t;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * Two way PCA.
 * <p>
 * Reference R implementation: <a href='http://models.life.ku.dk/sites/default/files/NPLS_Rver.zip>Download</a>
 * <p>
 *
 * @author Steven Lang
 */
public class TwoWayPCA extends UnsupervisedAlgorithm implements LoadingMatrixAccessor, Filter {

  private static final long serialVersionUID = -5392986328307308366L;

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(TwoWayPCA.class);

  protected int numComponents;

  /** (I x F) Matrix of scores/loadings of first order of X */
  protected INDArray T;

  /** Principal components of the fitted matrix */
  protected INDArray components;

  /** Means of the input matrix */
  protected INDArray xMeans;

  /** Explained variance */
  protected INDArray explainedVar;

  /** Whiten */
  protected boolean whiten;

  @Override
  protected void initialize() {
    super.initialize();
    numComponents = 3;
    whiten = false;
  }

  @Override
  protected void resetState() {
    super.resetState();
    T = null;
  }

  @Override
  protected String check(Tensor x) {
    String superCheck = super.check(x);
    if (superCheck != null){
      return superCheck;
    }
    return null;
  }

  @Override
  protected String doBuild(Tensor x) {
    INDArray X = x.getData();

    // Center data
    xMeans = X.mean(0);
    X = X.subRowVector(xMeans);

    // Define dimensions
    int I = (int) X.size(0);
    int nx = (int) X.size(1);

    final Map<String, INDArray> svd;
    final INDArrayIndex[] colSelect = {all(), interval(0, numComponents)};

    // Get components via SVD
    if (nx < I) {
      INDArray cov = (t(X).mmul(X)).div(I - 1);
      svd = MathUtils.svd(cov);
      components = svd.get("V").get(colSelect).dup();
    }
    else {
      INDArray cov = (X.mmul(t(X))).div(I - 1);
      svd = MathUtils.svd(cov);
      INDArray V = t(X).mmul(svd.get("V"));
      V = V.divRowVector(V.norm2(0));
      components = V.get(colSelect).dup();
    }

    final INDArray S = svd.get("SVAL").get(colSelect).dup();
    explainedVar = S.mul(S).div(I - 1);

    // Transform
    T = X.mmul(components);
    return null;
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
   * Get whether to whiten the input when filtering.
   *
   * @return True if whiten is enabled
   */
  public boolean isWhiten() {
    return whiten;
  }

  /**
   * Set whether to whiten the input when filtering.
   *
   * @param whiten True if whiten shall be enabled
   */
  public void setWhiten(boolean whiten) {
    this.whiten = whiten;
  }

  @Override
  protected Set<CriterionType> getAvailableStoppingCriteria() {
    return ImmutableSet.of();
  }

  @Override
  public Map<String, Tensor> getLoadingMatrices() {
    return ImmutableMap.of(
      "T", Tensor.create(T),
      "COMPONENTS", Tensor.create(components)
    );
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

    INDArray x = input.getData();

    // Center
    x = x.subRowVector(this.xMeans);

    INDArray xTransformed = x.mmul(t(components));

    // Whiten
    if (this.whiten) {
      xTransformed = xTransformed.divRowVector(Transforms.sqrt(this.explainedVar));
    }

    // Transform
    return Tensor.create(xTransformed);
  }
}
