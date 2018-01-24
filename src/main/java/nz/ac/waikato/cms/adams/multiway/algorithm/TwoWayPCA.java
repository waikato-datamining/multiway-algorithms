package nz.ac.waikato.cms.adams.multiway.algorithm;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.Filter;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.LoadingMatrixAccessor;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.UnsupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;

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

  @Override
  protected void initialize() {
    super.initialize();
    numComponents = 3;
  }

  @Override
  protected void resetState() {
    super.resetState();
    T = null;
  }

  @Override
  protected String check(Tensor x) {
    return null;
  }

  @Override
  protected String doBuild(Tensor x) {
    INDArray X = x.getData();

    // Center data
    xMeans = X.mean(0);
    X = X.divRowVector(xMeans);

    // Define dimensions
    int I = X.size(0);
    int nx = X.size(1);

    // Get components via SVD
    if (nx < I) {
      INDArray cov = (t(X).mmul(X)).div(I - 1);
      components = MathUtils.svd(cov).get("V").get(all(), interval(0, numComponents)).dup();
    }
    else {
      INDArray cov = (X.mmul(t(X))).div(I - 1);
      INDArray v = t(X).mmul(MathUtils.svd(cov).get("V"));
      v = v.divRowVector(v.norm2(0));
      components = v.get(all(), interval(0, numComponents)).dup();
    }

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
    INDArray x = input.getData();

    // Center
    x = x.divRowVector(this.xMeans);

    // Transform
    return Tensor.create(x.mmul(t(components)));
  }
}
