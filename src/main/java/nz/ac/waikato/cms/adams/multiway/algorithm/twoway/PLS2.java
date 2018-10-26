package nz.ac.waikato.cms.adams.multiway.algorithm.twoway;

import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.Filter;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.SupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.Criterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.exceptions.InvalidInputException;
import nz.ac.waikato.cms.adams.multiway.exceptions.ModelNotBuiltException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Set;

import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.concat;

/**
 * PLS2 Algorithm implementation.
 *
 * Implementation according to <a href="https://web.archive.org/web/20160717065734/http://statmaster.sdu.dk:80/courses/ST02/module08/module.pdf">Partial least squares regression II</a>
 *
 * @author Steven Lang
 */
public class PLS2 extends SupervisedAlgorithm implements Filter {

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(PLS2.class);

  private static final long serialVersionUID = -2509287702659489326L;

  /** Number of components */
  protected int numComponents;

  /** Matrix of scores of Y */
  protected INDArray U;

  /** Matrix of scores of X */
  protected INDArray T;

  /** Loadings of X */
  protected INDArray W;

  /** (Jy x F) Loadings in second order of Y */
  protected INDArray Q;

  /** Regression coefficient of MLR of X on t */
  protected INDArray P;

  /** Regression coefficient of MLR of u on t */
  protected INDArray C;

  /** Mean vector of the target matrix */
  protected INDArray yMean;

  /** Std vector of the target matrix */
  protected INDArray yStd;

  /** Mean vector of the input matrix */
  protected INDArray xMean;

  /** Std vector of the input matrix */
  protected INDArray xStd;

  /** Whether to standardize Y or not */
  protected boolean standardizeY;

  /** Whether to calculate w in inner loop or not */
  protected boolean wStepInnerLoop;

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
      resetState();
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
    resetState();
  }

  @Override
  protected void initialize() {
    super.initialize();
    numComponents = 5;
    standardizeY = true;
    wStepInnerLoop = true;
    addStoppingCriterion(CriterionUtils.iterations(250));
    addStoppingCriterion(CriterionUtils.improvement(1e-7));
  }

  @Override
  protected String doBuild(Tensor x, Tensor y) {
    INDArray X = x.getData();
    INDArray Y = y.getData();


    if (X.rank() == 3) {
      X = MathUtils.matricize(X, 0);
    }

    xMean = X.mean(0);
    yMean = Y.mean(0);
    xStd = X.std(0);
    yStd = Y.std(0);

    // Center X and Y
    X = X.subRowVector(xMean);
    Y = Y.subRowVector(yMean);

    // Standardize Y if necessary
    if (standardizeY) {
      Y = Y.divRowVector(yStd);
    }

    INDArray Xres = X.dup();
    INDArray Yres = Y.dup();
    INDArray t = null;
    INDArray q = null;
    INDArray w = null;
    INDArray p;
    INDArray c;
    INDArray u = Y.getColumn(0);
    INDArray uOld;

    // Loop over components
    for (int j = 0; j < numComponents && !stoppingCriteriaMatch(); j++) {

      // If the w step should not be computed in the loop, do it here
      if (!wStepInnerLoop){
        // w step
        w = calcW(Xres, Yres, u, j);
      }

      while (!stoppingCriteriaMatch()) {

        // If the w step should be computed in the loop, do it here
	if (wStepInnerLoop){
          // w step
          w = calcW(Xres, Yres, u, j);
        }

        // t step
	t = Xres.mmul(w);

	// q step
	q = Yres.transpose().mmul(t);
	q = q.div(q.norm2());

	// u step
	uOld = u;
	u = Yres.mmul(q);

	updateUStoppingCriteria(uOld, u);
      }
      resetStoppingCriteria();

      // It might happen, that the for loop was entered but the while loop was
      // skipped (due to force stop). If this is the case: t,q and u will be
      // null - so check for forceStop here
      if (isForceStop()){
        return "Algorithm force stopped.";
      }

      // c and p step
      INDArray tmmult = t.transpose().mmul(t);
      c = t.transpose().mmul(u).div(tmmult);
      p = Xres.transpose().mmul(t).div(tmmult);

      // Deflate X and Y
      Xres = Xres.sub(t.mmul(p.transpose()));
      Yres = Yres.sub(t.mmul(q.transpose()).mul(c));

      // Store vectors
      P = concat(P, p, 1);
      Q = concat(Q, q, 1);
      T = concat(T, t, 1);
      U = concat(U, u, 1);
      C = concat(C, c, 1);
      W = concat(W, w, 1);
    }

    if (isForceStop()){
      return "Algorithm force stopped.";
    }

    // Transform C to a diagonale matrix
    C = fromDiagVector(C);

    return null;
  }

  /**
   * Calculate direction vector w
   * @param xres X residuals
   * @param u Score y
   * @param j Current component index
   * @return Direction vector w
   */
  protected INDArray calcW(INDArray xres, INDArray yres, INDArray u, int j) {
    INDArray w;
    w = xres.transpose().mmul(u);
    w = w.div(w.norm2());
    return w;
  }


  @Override
  public Tensor predict(Tensor x) {
    if (x.getData().rank() == 3) {
      x = Tensor.create(MathUtils.matricize(x.getData(), 0));
    }

    INDArray Bhat = W.mmul(MathUtils.invert(P.transpose().mmul(W))).mmul(C).mmul(Q.transpose());

    INDArray X = x.getData();
    INDArray Xcentered = X.subRowVector(xMean);
    INDArray Yhat = Xcentered.mmul(Bhat);

    if (standardizeY) {
      Yhat.muliRowVector(yStd);
    }
    Yhat.addiRowVector(yMean);

    return Tensor.create(Yhat);
  }

  @Override
  public Tensor filter(Tensor input) {
    // Check if the model has been built yet
    if (!isFinished()) {
      throw new ModelNotBuiltException(
	"Trying to invoke filter(Tensor input) while the model has not been " +
	  "built yet."
      );
    }

    INDArray T = input.getData().mmul(W.mmul(MathUtils.invert(P.transpose().mmul(W))));
    return Tensor.create(T);
  }

  /**
   * Create a diagonale matrix from a vector
   *
   * @param vec Vector which is to be set as diagonale of the output matrix
   * @return Diagonale matrix from vector
   */
  protected INDArray fromDiagVector(INDArray vec) {
    if (!vec.isVector()) {
      throw new InvalidInputException("Input was not a vector.");
    }
    long n = vec.length();
    INDArray mat = Nd4j.eye(n);
    return mat.mulRowVector(vec);
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
   * Calculate the improvement of the u vector.
   *
   * @param uold Old u
   * @param unew New u
   * @return L2 distance from uold to unew
   */
  protected double calcUimprovement(INDArray uold, INDArray unew) {
    return uold.distance2(unew);
  }

  /**
   * Update the internal state.
   */
  protected boolean updateUStoppingCriteria(INDArray uold, INDArray unew) {
    // Update stopping criteria states
    for (Criterion sc : stoppingCriteria.values()) {
      switch (sc.getType()) {
	case IMPROVEMENT:
	  sc.update(calcUimprovement(uold, unew));
	  break;
	default:
	  sc.update();
      }
    }

    return true;
  }
}
