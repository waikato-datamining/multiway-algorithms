package com.github.waikatodatamining.multiway.data;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

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
      final INDArray inv = invert(arr.transpose().mmul(arr), inPlace).mmul(arr.transpose());
      if (inPlace)
	arr.assign(inv);
      return inv;
    }
    catch (SingularMatrixException e) {
      return pseudoInvert(arr, inPlace);
    }
  }

  /**
   * Build the inverse of a matrix
   *
   * @param arr the array to invert
   * @return the inverted matrix
   */
  public static INDArray invert(INDArray arr, boolean inPlace) {
    if (arr.columns() != arr.rows()) {
      throw new IllegalArgumentException("invalid array: must be square matrix");
    }
    RealMatrix rm = CheckUtil.convertToApacheMatrix(arr);
    RealMatrix rmInverse = new LUDecomposition(rm).getSolver().getInverse();


    INDArray inverse = CheckUtil.convertFromApacheMatrix(rmInverse);
    if (inPlace)
      arr.assign(inverse);
    return inverse;
  }

  /**
   * Short handle for transposing an array
   *
   * @param arr Array to transpose
   * @return Transposed array
   */
  public static INDArray t(INDArray arr) {
    return arr.transpose();
  }

  /**
   * Build the inverse of a matrix
   *
   * @param arr the array to invert
   * @return the inverted matrix
   */
  public static INDArray invert(INDArray arr) {
    return invert(arr, false);
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
   * Revert {@link MathUtils#matricize(INDArray, int)}
   *
   * @param X    Input array
   * @param axis Fold axis
   * @param dim2 First reshape dimension
   * @param dim3 Second reshape dimension
   * @return Folded Tensor
   */
  public static INDArray invertMatricize(INDArray X, int axis, int dim2, int dim3) {
    // Convert negative axis to equivalent positive form
    final int dims = 3;
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

    int[] invPerm = new int[dims];
    for (int i = 0; i < dims; i++) {
      invPerm[permutation[i]] = i;
    }

    X = X.reshape(X.size(0), dim3, dim2);
    X = X.permute(invPerm);
    return X;
  }

  /**
   * Invert vectorization of a Matrix.
   *
   * @param x    input vector
   * @param dim1 first dimension of the matrix
   * @return Matricized vector
   */
  public static INDArray invertVectorize(INDArray x, int dim1) {
    return x.reshape(-1, dim1).permute(1, 0);
  }

  /**
   * Converts a Nd4j INDArray into a double matrix
   *
   * @param arr INDArray
   * @return double matrix
   */
  public static double[][] to2dDoubleArray(INDArray arr) {
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

  /**
   * Converts a Nd4j INDArray into a double matrix
   *
   * @param arr INDArray
   * @return double matrix
   */
  public static double[][][] to3dDoubleArray(INDArray arr) {
    if (arr.shape().length != 3) {
      throw new RuntimeException("Matrix must be three-dimensional.");
    }
    double[][][] res = new double[arr.size(0)][arr.size(1)][arr.size(2)];
    for (int i = 0; i < arr.size(0); i++) {
      for (int j = 0; j < arr.size(1); j++) {
	for (int k = 0; k < arr.size(2); k++) {
	  res[i][j][k] = arr.getDouble(i, j, k);
	}
      }
    }

    return res;
  }

  /**
   * Read an INDArray from a 3d double matrix.
   *
   * @param data Input data
   * @return Data represented as INDArray
   */
  public static INDArray from3dDoubleArray(double[][][] data) {
    int numRows = data.length;
    int numColumns = data[0].length;
    int numDimensions = data[0][0].length;
    // Create array of slices
    INDArray X = Nd4j.create(numRows, numColumns, numDimensions);
    for (int i = 0; i < data.length; i++) {
      double[][] slice = data[i];
      X.putRow(i, Nd4j.create(slice));
    }
    return X;
  }

  /**
   * Centers the array along a given axis.
   *
   * @param arr  Array to be centered
   * @param axis Center axis
   * @return Centered array
   */
  public static INDArray standardize(INDArray arr, int axis) {
    INDArray unfolded = matricize(arr, axis);

    // Center across first axis
    final INDArray sumsFirstAxis = unfolded.sum(0);
    final INDArray meansFirstAxis = sumsFirstAxis.div(unfolded.size(0));
    final INDArray res = unfolded.subRowVector(meansFirstAxis);

    // Scale across second axis
    final INDArray squared = unfolded.mul(unfolded);
    final INDArray sumsSquaredSecondAxis = squared.sum(1);
    final INDArray sqrtSumsSquaredSecondAxis = Transforms.sqrt(sumsSquaredSecondAxis);

    res.diviColumnVector(sqrtSumsSquaredSecondAxis);

    int dim1 = 0;
    int dim2 = 0;
    if (axis == 0) {
      dim1 = 1;
      dim2 = 2;
    }
    else if (axis == 1) {
      dim1 = 0;
      dim2 = 2;
    }
    else if (axis == 2) {
      dim1 = 0;
      dim2 = 1;
    }
    return invertMatricize(res, axis, arr.size(dim1), arr.size(dim2));
  }

  /**
   * Centers the array along a given axis.
   *
   * @param arr  Array to be centered
   * @param axis Center axis
   * @return Centered array
   */
  public static INDArray center(INDArray arr, int axis) {
    // Center across first axis
    final INDArray sumsFirstAxis = arr.sum(axis);
    final INDArray meansFirstAxis = sumsFirstAxis.div(arr.size(axis));
    return arr.sub(meansFirstAxis.broadcast(arr.shape()));
  }

  /**
   * Centers the array along a given axis.
   *
   * @param arr  Array to be centered
   * @param axis Center axis
   * @return Centered array
   */
  public static double[][][] standardize(double[][][] arr, int axis) {
    return to3dDoubleArray(standardize(from3dDoubleArray(arr), axis));
  }
}
