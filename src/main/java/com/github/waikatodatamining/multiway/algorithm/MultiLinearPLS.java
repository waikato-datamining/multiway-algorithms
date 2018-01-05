package com.github.waikatodatamining.multiway.algorithm;

import com.github.waikatodatamining.multiway.algorithm.stopping.CriterionType;
import com.github.waikatodatamining.multiway.data.MathUtils;
import com.github.waikatodatamining.multiway.exceptions.InvalidInputException;
import com.google.common.collect.ImmutableSet;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Set;

public class MultiLinearPLS extends AbstractAlgorithm {

  private int numComponents;

  public MultiLinearPLS(int numComponents) {
    super();
    this.numComponents = numComponents;
  }

  public void buildModel(double[][][] independent, double[] dependent) {
    validateInput(independent);

    final int numRows = independent.length;
    final int numColumns = independent[0].length;
    final int numDimensions = independent[0][0].length;

    INDArray X = MathUtils.from3dDoubleArray(independent);
    INDArray y = Nd4j.create(dependent).transpose();
    INDArray Xunfolded = MathUtils.matricize(X, 0);
    final INDArray w = getW(Xunfolded, y);
    final INDArray t = Xunfolded.mmul(w);
    INDArray X1 = Xunfolded.sub(t.mmul(w.transpose()));

    INDArray b1 = MathUtils.invert(t.transpose().mmul(t),false).mmul(t.transpose()).mmul(y);
    INDArray y2 = y.sub(t.mmul(b1));


  }

  protected INDArray getW(INDArray X, INDArray y) {
    final INDArray[] wjwk = getWjWk(X, y);
    return MathUtils.outer(wjwk[1], wjwk[0]);
  }

  protected INDArray[] getWjWk(INDArray X, INDArray y) {
    // z_jk = sum_i { y_i * x_ijk }
    INDArray Z = X.mulColumnVector(y).sum(0); // Is that correct?

    // Solve SVD
    SingularValueDecomposition svd = new SingularValueDecomposition(CheckUtil.convertToApacheMatrix(Z));
    final INDArray U = CheckUtil.convertFromApacheMatrix(svd.getU());
    final INDArray V = CheckUtil.convertFromApacheMatrix(svd.getV());

    // w^J and w^K are the first left and the first right singular vectors
    final INDArray wJ = U.getColumn(0);
    final INDArray wK = V.getColumn(0);

    return new INDArray[]{wJ, wK};
  }

  @Override
  protected void update() {

  }

  @Override
  protected Set<CriterionType> getAvailableStoppingCriteria() {
    return ImmutableSet.of();
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
}

