package com.github.waikatodatamining.multiway.data;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.inverse.InvertMatrix;

/**
 * Math utilities.
 *
 * @author Steven Lang
 */

public class MathUtils {

  /**
   * Build the pseudo inverse of a matrix
   *
   * @param arr the array to invert
   * @return the inverted matrix
   */
  public static INDArray pseudoInvert(INDArray arr, boolean inPlace) {

    RealMatrix rm = CheckUtil.convertToApacheMatrix(arr);
    RealMatrix rmInverse = new SingularValueDecomposition(rm).getSolver().getInverse();


    INDArray inverse = CheckUtil.convertFromApacheMatrix(rmInverse);
    if (inPlace)
      arr.assign(inverse);
    return inverse;
  }

  /**
   * Build the pseudo inverse of a matrix
   *
   * @param arr the array to invert
   * @return the inverted matrix
   */
  public static INDArray pseudoInvert2(INDArray arr, boolean inPlace) {
    try {
      final INDArray inv = InvertMatrix.invert(arr.transpose().mmul(arr), inPlace).mmul(arr.transpose());
      if (inPlace)
	arr.assign(inv);
      return inv;
    }
    catch (SingularMatrixException e) {
      return pseudoInvert(arr, inPlace);
    }
  }

  /**
   * Calculate the column wise Khatri-Rao product.
   *
   * @param U Left hand side matrix
   * @param V Right hand side matrix
   * @return Column wise Khatri-Rao product
   * @see <a href="https://en.wikipedia.org/wiki/Kronecker_product#Khatri–Rao_product"/>
   */
  public static INDArray khatriRaoProductColumnWise(INDArray U, INDArray V) {
    // Assume U.size(1) == V.size(1)
    if (U.size(1) != V.size(1)) {
      throw new RuntimeException("U and V did not match in column dimension.");
    }

    // Check same dimensions
    if (U.shape().length != V.shape().length) {
      throw new RuntimeException("dim(U) != dim(V). Dimension mismatch");
    }
    final int dim = U.size(1);
    INDArray res = Nd4j.create(U.size(0) * V.size(0), dim);
    for (int i = 0; i < dim; i++) {
      final INDArray ui = U.get(NDArrayIndex.all(), NDArrayIndex.point(i)).dup();
      final INDArray vi = V.get(NDArrayIndex.all(), NDArrayIndex.point(i)).dup();

      // Build outer product: ui (x) vi and reshape into a column vector
      final INDArray krUiVi = outer(ui, vi);
      final INDArray slicei = krUiVi.reshape(ui.size(0) * vi.size(0), 1);
      res.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(i)}, slicei);
    }

    return res;
  }

  /**
   * Outer product
   *
   * @param x Left product argument
   * @param y Right product argument
   * @return Outer product
   */
  public static INDArray outer(INDArray x, INDArray y) {
    final INDArray ytrans = y.transpose();
    final INDArray res = x.mmul(ytrans);
    return res.reshape('F', res.size(0), res.size(1));
  }

  /**
   * Returns flattened version of tensor.
   * See http://www.graphanalysis.org/SIAM-PP08/Dunlavy.pdf
   * for more details.
   *
   * @param X    Input tensor
   * @param axis Flattening axis
   * @return Flattened tensor
   */
  public static INDArray matricize(INDArray X, int axis) {

    // Convert negative axis to equivalent positive form
    final int dims = X.shape().length;
    if (axis < 0) {
      axis = dims + axis;
    }

    // Collect possible axes
    int[] axes = new int[dims];
    for (int i = 0; i < dims; i++) {
      axes[dims - 1 - i] = i;
    }


    // Generate permutation array
    final int[] rms = ArrayUtils.removeElement(axes, axis);
    int[] permutation = new int[dims];
    permutation[0] = axis;
    for (int i = 0; i < rms.length; i++) {
      permutation[i + 1] = rms[i];
    }

    final INDArray perm = X.permute(permutation);
    return perm.reshape(X.size(axis), -1);
  }

  /**
   * Converts a Nd4j INDArray into a double matrix
   *
   * @param arr INDArray
   * @return double matrix
   */
  public static double[][] toDoubleMatrix(INDArray arr) {
    if (arr.shape().length != 2) {
      throw new RuntimeException("Matrix must be two-dimensional.");
    }
    double[][] res = new double[arr.size(0)][arr.size(1)];
    for (int i = 0; i < arr.size(0); i++) {
      for (int j = 0; j < arr.size(1); j++) {
	res[i][j] = arr.getDouble(i, j);
      }
    }

    return res;
  }
}
