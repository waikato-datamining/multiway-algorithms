package nz.ac.waikato.cms.adams.multiway.algorithm.twoway;

import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.Criterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Set;

import static nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType.IMPROVEMENT;
import static nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType.ITERATION;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * Mixed-Norm PLS algorithm.
 * <p>
 * Implementation according to <a href="https://www.sciencedirect.com/science/article/abs/pii/S0169743916000058">Mixed-norm
 * partial least squares</a>
 *
 * @author Steven Lang
 */
public class MNPLS extends PLS2 {

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(MNPLS.class);

  private static final long serialVersionUID = -2509287702659489326L;

  /** Epsilon */
  protected static double EPS = 1e-6;


  @Override
  protected void initialize() {
    super.initialize();
    wStepInnerLoop = false;
    addStoppingCriterion(CriterionUtils.improvement(1e-4));
  }

  @Override
  protected INDArray calcW(INDArray xres, INDArray yres, INDArray u, int j) {
    INDArray W = getW(xres, yres);
    return W.getColumn(j);
  }

  /**
   * Initialize W according to the MNPLS algorithm.
   *
   * @param x Input X
   * @param y Target Y
   */
  protected INDArray getW(INDArray x, INDArray y) {
    INDArray W;
    INDArray D;
    double alpha;
    double beta;
    INDArray Wold;
    int seed = 0;

    // Initialize W, D, alpha, beta
    W = normalized(Nd4j.randn(x.size(1), numComponents, seed));
    D = calcD(W);
    alpha = calcAlpha(W, D);
    beta = calcBeta(W, D, x, y);
    final Criterion iterCrit =  stoppingCriteria.get(ITERATION).copy();
    final Criterion impCrit =  stoppingCriteria.get(IMPROVEMENT).copy();
    while (!isForceStop() && !iterCrit.matches() && !impCrit.matches()) {
      // Update W
      Wold = W;
      W = calcW(x, y, D, alpha, beta);

      // Update alpha and beta
      alpha = calcAlpha(W, D);
      beta = calcBeta(W, D, x, y);

      // Update D
      D = calcD(W);

      // Update stopping criteria
      double improvement = calcWImprovement(Wold, W);
      impCrit.update(improvement);
      iterCrit.update();
    }

    return W;
  }

  /**
   * Normalize A along axis 0.
   *
   * @param A Input matrix
   * @return Normalized input
   */
  protected INDArray normalized(INDArray A) {
    INDArray columnNorms = A.norm2(0);
    return A.divRowVector(columnNorms);
  }

  /**
   * Get the diagonale of a matrix as vector.
   *
   * @param A Input matrix
   * @return Diagonale of {@code A} as vector
   */
  protected INDArray getDiagVector(INDArray A) {
    double[] diag = new double[(int) A.size(0)];
    for (int i = 0; i < A.size(0); i++) {
      diag[i] = A.getDouble(i, i);
    }
    return Nd4j.create(diag);
  }

  /**
   * Calculate the trace of a matrix.
   *
   * @param A Input matrix
   * @return Trace of {@code A}
   */
  protected double trace(INDArray A) {
    return getDiagVector(A).sum().getDouble(0);
  }

  /**
   * Initialize the alpha value.
   *
   * @return Alpha
   */
  protected double calcAlpha(INDArray W, INDArray D) {
    return 2.0 / trace(W.transpose().mmul(D).mmul(W));
  }

  /**
   * Initialize the beta value.
   *
   * @return Beta
   */
  protected double calcBeta(INDArray W, INDArray D, INDArray X, INDArray Y) {
    double a = 2.0 * trace(W.transpose().mmul(X.transpose()).mmul(Y).mmul(Y.transpose()).mmul(X).mmul(W));
    double b = trace(W.transpose().mmul(D).mmul(W));
    return a / (b * b);
  }

  /**
   * Calculate the new W matrix.
   *
   * @return Updated W matrix
   */
  protected INDArray calcW(INDArray X, INDArray Y, INDArray D, double alpha, double beta) {
    INDArray A = X.transpose().mul(alpha).mmul(Y).mmul(Y.transpose()).mmul(X).sub(D.mul(beta));
    INDArray B = X.transpose().mmul(X);

    INDArray vecs = MathUtils.generalizedEigenvectors(A, B);
    INDArray firstKvecs = vecs.get(all(), interval(0, numComponents));
    return firstKvecs;
  }

  /**
   * Calculate the new D matrix.
   *
   * @param W W matrix (loadings)
   * @return Updated D matrix
   */
  protected INDArray calcD(INDArray W) {
    INDArray wnorm2 = W.norm2(1);
    for (int i = 0; i < wnorm2.size(0); i++) {
      if (StrictMath.abs(wnorm2.getDouble(i)) < EPS) {
	wnorm2.putScalar(i, EPS);
      }
    }

    INDArray ones = Nd4j.ones(W.size(0));
    INDArray vec = ones.div(wnorm2.mul(2.0));

    return fromDiagVector(vec);
  }

  /**
   * Calculate the W matrix improvement.
   *
   * @param Wold Old W
   * @param Wnew New W
   * @return Square root of the trace of frobenius norm of {@code Wold - Wnew}
   */
  private double calcWImprovement(INDArray Wold, INDArray Wnew) {
    INDArray diff = Wold.sub(Wnew);
    return StrictMath.sqrt(trace(diff.mmul(diff.transpose())));
  }

  @Override
  protected Set<CriterionType> getAvailableStoppingCriteria() {
    return ImmutableSet.of(
      CriterionType.IMPROVEMENT,
      CriterionType.ITERATION,
      CriterionType.TIME
    );
  }

  /**
   * Update the internal state.
   */
  protected void updateWcalcStoppingCriteria(INDArray Wold, INDArray Wnew) {
    // Update stopping criteria states
    for (Criterion sc : stoppingCriteria.values()) {
      switch (sc.getType()) {
	case IMPROVEMENT:
	  sc.update(calcWImprovement(Wold, Wnew));
	  break;
	default:
	  sc.update();
      }
    }
  }
}
