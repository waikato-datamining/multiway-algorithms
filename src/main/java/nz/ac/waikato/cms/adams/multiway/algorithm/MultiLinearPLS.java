package nz.ac.waikato.cms.adams.multiway.algorithm;

import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.exceptions.InvalidInputException;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Set;

import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.center;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.from3dDoubleArray;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.invert;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.invertVectorize;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.matricize;
import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.t;

/**
 * Multilinear Partial Least Squares Regression.
 * <p>
 * Implementation according to <a href='http://onlinelibrary.wiley.com/doi/10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C/epdf'>R. Bro Multiway Calibration, Multilinear PLS</a>
 * <p>
 *
 * @author Steven Lang
 */
public class MultiLinearPLS extends AbstractAlgorithm {

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(MultiLinearPLS.class);


  /** Serial version UID */
  private static final long serialVersionUID = 6121171390172636096L;

  /** Number of components / PLS iterations */
  protected int numComponents;

  /** Weights W */
  protected INDArray W;

  /** Weights ba */
  protected INDArray ba;

  /** Weights bPLS for predictions */
  protected INDArray bPLS;

  /**
   * Build the internal model.
   *
   * @param independent Independent variables (X)
   * @param dependent   Dependent variables (y)
   */
  public void buildModel(double[][][] independent, double[] dependent) {
    // TODO: Center X and y or add column of [1] to T
    // TODO: validate dependent as well
    validateInput(independent, dependent);

    final int numRows = independent.length;
    final int numColumns = independent[0].length;
    final int numDimensions = independent[0][0].length;

    INDArray X = from3dDoubleArray(independent);
    // Center X across the first mode

    INDArray y = t(Nd4j.create(dependent));
    y = y.subRowVector(y.sum(0).div(y.size(0)));
    INDArray Xa = matricize(X, 0);
    Xa = center(Xa, 0);
    INDArray ya = y;
    INDArray T = null;
    //    INDArray T = Nd4j.ones(numRows, 1);
    INDArray ta;
    INDArray wa;
    for (int a = 0; a < numComponents; a++) {
      wa = getW(Xa, ya, numColumns, numDimensions);  // maybe y_0 instead of y_a?
      double wTw = t(wa).mmul(wa).getDouble(0);
      if (Math.abs(1 - wTw) > 1E-5) {
	throw new RuntimeException("Condition w^T*w = 1 violated, (was " + wTw + ")");
      }
      //      log.debug(wa.toString());
      W = concat(W, wa, 1);
      //      log.debug(W.toString());
      ta = Xa.mmul(wa);
      //      log.debug(ta.toString());
      T = concat(T, ta, 1);
      //      log.debug(T.toString());
      Xa = Xa.sub(ta.mmul(t(wa)));
      ba = invert(t(T).mmul(T)).mmul(t(T)).mmul(ya); // maybe y_0 instead of y_a?
      //      log.debug(ba.toString());
      ya = ya.sub(T.mmul(ba)); // maybe y_0 instead of y_a?
      //      log.debug(ya.toString());
    }

    log.debug("y = " + y);
    log.debug("ya = " + ya);
    log.debug("T = \n " + T);
    log.debug("ba = " + ba);
    log.debug("T.mmul(ba) - ya= " + T.mmul(ba).sub(ya));

    INDArray[] bPlsNoWa = new INDArray[numComponents];
    final INDArray I = Nd4j.zeros(numDimensions * numColumns, numDimensions * numColumns);
    for (int i = 0; i < numComponents; i++) {
      I.putScalar(i, i, 1);
    }
    bPlsNoWa[0] = I.dup();

    for (int i = 1; i < numComponents; i++) {
      INDArray lastEntry = bPlsNoWa[i - 1];
      final INDArray lastwi = W.getColumn(i - 1).dup();
      final INDArray IsubwwT = I.sub(lastwi.mmul(t(lastwi)));
      bPlsNoWa[i] = lastEntry.mmul(IsubwwT);
    }

    INDArray blpls0 = Nd4j.create(numDimensions * numColumns, numComponents);
    for (int i = 0; i < numComponents; i++) {
      blpls0.putColumn(i, bPlsNoWa[i].mmul(W.getColumn(i)));
    }
    bPLS = blpls0.mmul(ba);
  }

  /**
   * Predict the dependent variable for a new set of inputs.
   *
   * @param data New data
   * @return Predicted dependent variables y for the given data
   */
  public double[] predict(double[][][] data) {
    final INDArray X = from3dDoubleArray(data);
    double[] preds = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      INDArray Xi = X.getRow(i).reshape(1, X.size(1), X.size(2));
      INDArray XiMatricized = matricize(Xi, 0);
      final INDArray mmul = XiMatricized.mmul(bPLS);
      preds[i] = mmul.getDouble(0);
    }
    return preds;
  }

  /**
   * Concatenate a matrix and a vector at a given axis. Still valid if {@code U}
   * is {@code null}, as {@code U} will then be initialized with {@code ua} as
   * first vector at the given axis {@code axis}.
   *
   * @param U    Matrix
   * @param ua   Vector
   * @param axis Concatenation axis
   * @return Concatenation of the matrix and the vector at the given axis
   */
  protected INDArray concat(INDArray U, INDArray ua, int axis) {
    final INDArray uaReshaped = ua.reshape(-1, 1);
    if (U == null) {
      return uaReshaped;
    }
    else {
      return Nd4j.concat(axis, U, uaReshaped);
    }
  }

  /**
   * Compute w as
   * w = kronecker(w^J,w^K)
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
  protected INDArray getW(INDArray X, INDArray y, int numColumns, int numDimensions) {
    final INDArray[] wjwk = getWjWk(X, y, numColumns);
    final INDArray kronecker = MathUtils.outer(wjwk[1], wjwk[0]);
    return kronecker.reshape(numColumns * numDimensions, -1);
  }


  /**
   * Compute  (w^J,w^K) = SVD(Z) with Vec(Z) = X^T*y
   * w^J: first left singularvector of SVD(Z)
   * w^K: first right singularvector of SVD(Z)
   *
   * @param X          Matricized input
   * @param y          Dependent variables
   * @param numColumns Number of columns
   * @return w^J, w^K
   */
  protected INDArray[] getWjWk(INDArray X, INDArray y, int numColumns) {
    // z_jk = sum_i { y_i * x_ijk }
    INDArray Z = invertVectorize(t(X).mmul(y), numColumns);

    // Solve SVD
    SingularValueDecomposition svd = new SingularValueDecomposition(CheckUtil.convertToApacheMatrix(Z));
    final INDArray U = CheckUtil.convertFromApacheMatrix(svd.getU());
    final INDArray V = CheckUtil.convertFromApacheMatrix(svd.getV());

    // w^J and w^K are the first left and the first right singular vectors
    INDArray wJ = U.getColumn(0).dup();
    INDArray wK = V.getColumn(0).dup();

    // normalize w^J and w^K
    wJ = Transforms.unitVec(wJ);
    wK = Transforms.unitVec(wK);

    return new INDArray[]{wJ, wK};
  }


  @Override
  protected Set<CriterionType> getAvailableStoppingCriteria() {
    return ImmutableSet.of();
  }

  /**
   * Validate the input data
   *
   * @param inputMatrix Input data
   * @param y           Dependent variables
   */
  protected void validateInput(double[][][] inputMatrix, double[] y) {
    if (inputMatrix.length == 0
      || inputMatrix[0].length == 0
      || inputMatrix[0][0].length == 0) {
      throw new InvalidInputException("Input matrix dimensions must be " +
	"greater than 0.");
    }

    if (inputMatrix.length != y.length) {
      throw new InvalidInputException("Independent and dependent variables must" +
	" be of the same length.");
    }
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
}

