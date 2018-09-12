package nz.ac.waikato.cms.adams.multiway.data;

import com.google.common.collect.ImmutableMap;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.exceptions.InvalidInputException;
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

import java.util.Map;

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
   * @see <a href="https://en.wikipedia.org/wiki/Kronecker_product#Khatri–Rao_product">Wikipedia</a>
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
    final long dim = U.size(1);
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
    int xSize = x.shape().length;
    int ySize = y.shape().length;
    long[] xShape = x.shape();
    long[] yShape = y.shape();

    // Extend shape: [a,b,c] -> [a,b,c,1]
    if (x.size(xSize - 1 ) != 1){
      xShape = extendIdxToArray(xShape, 1);
      xSize = xShape.length;
      x = x.reshape(xShape);
    }

    // Extend shape: [a,b,c] -> [a,b,c,1]
    if (y.size(ySize -1 ) != 1){
      yShape = extendIdxToArray(yShape, 1);
      ySize =  yShape.length;
      y = y.reshape(yShape);
    }

    return Nd4j.tensorMmul(x, y, new int[][]{{xSize -1 },{ySize - 1}});
  }


  public static long[] extendIdxToArray(long[] arr, long val){
    return extendIdxToArray(arr, arr.length, val);
  }
  
  public static long[] extendIdxToArray(long[] arr, int idx, long val){
    long[] extArr = new long[arr.length + 1];
    System.arraycopy(arr, 0, extArr, 0, idx);
    extArr[idx] = val;
    if (idx < arr.length){
      System.arraycopy(arr, idx, extArr, idx + 1, arr.length - idx);
    }

    return extArr;
  }

  public static long[] removeIdxFromArray(long[] arr, int idx){
    long[] redArr = new long[arr.length - 1];
    System.arraycopy(arr, 0, redArr, 0, idx);
    System.arraycopy(arr, idx + 1, redArr, idx, redArr.length - idx);
    return redArr;
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
    double[][] res = new double[(int) arr.size(0)][(int) arr.size(1)];
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
    double[][][] res = new double[(int) arr.size(0)][(int) arr.size(1)][(int) arr.size(2)];
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
    return invertMatricize(res, axis, (int)arr.size(dim1), (int)arr.size(dim2));
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
    final INDArray sums = arr.sum(axis);
    final INDArray means = sums.div(arr.size(axis));
    return arr.sub(means.broadcast(arr.shape()));
  }

  /**
   * Centers the array along a given axis.
   *
   * @param x    Array to be centered
   * @param axis Center axis
   * @return Centered array
   */
  public static Tensor center(Tensor x, int axis) {
    // Center across first axis
    INDArray arr = x.getData();
    final INDArray sums = arr.sum(axis);
    final INDArray means = sums.div(arr.size(axis));
    return Tensor.create(arr.sub(means.broadcast(arr.shape())));
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
  public static INDArray concat(INDArray U, INDArray ua, int axis) {
    final INDArray uaReshaped = ua.reshape(-1, 1);
    if (U == null) {
      return uaReshaped;
    }
    else {
      return Nd4j.concat(axis, U, uaReshaped);
    }
  }

  /**
   * Calculate the mean squared error between two tensors.
   *
   * @param a First tensor
   * @param b Second tensor
   * @return Mean Squared distance
   */
  public static double meanSquaredError(INDArray a, INDArray b) {
    return a.squaredDistance(b) / a.size(0);
  }

  /**
   * Calculate the mean squared error between two tensors.
   *
   * @param a First tensor
   * @param b Second tensor
   * @return Mean Squared distance
   */
  public static double meanSquaredError(Tensor a, Tensor b) {
    return meanSquaredError(a.getData(), b.getData());
  }

  /**
   * Perform an SVD on the given matrix.
   *
   * @param x Input matrix
   * @return SVD matrices U,S,V
   */
  public static Map<String, INDArray> svd(INDArray x) {
    final RealMatrix xApache = CheckUtil.convertToApacheMatrix(x);
    SingularValueDecomposition svd = new SingularValueDecomposition(xApache);
    final double[] singularValues = svd.getSingularValues();
    return ImmutableMap.of(
      "U", CheckUtil.convertFromApacheMatrix(svd.getU()),
      "S", CheckUtil.convertFromApacheMatrix(svd.getS()),
      "V", CheckUtil.convertFromApacheMatrix(svd.getV()),
      "SVAL", Nd4j.create(singularValues).transpose()
    );
  }


  /**
   * Orthonormalize the given matrix with the Gram-Schmidt process.
   *
   * @param V Input 2d matrix
   * @return Orthonormalized input matrix
   */
  public static INDArray orth(INDArray V, boolean normalize) {
    // Make sure V is a matrix
    if (V.shape().length != 2) {
      throw new InvalidInputException(
	String.format("Cannot orthogonalize tensors of with order != 2. Order " +
	  "was %d.", V.shape().length)
      );
    }

    INDArray U = Nd4j.create(V.shape());
    INDArray vi;
    INDArray ui;
    for (int i = 0; i < V.size(1); i++) {
      vi = V.getColumn(i);
      ui = vi.dup();
      for (int j = 0; j < i; j++) {
	ui = ui.sub(project(U.getColumn(j), vi));
      }
      U.putColumn(i, ui);
    }

    // Normalize
    if (normalize) {
      U = U.divRowVector(U.norm2(0));
    }
    return U;

  }

  /**
   * Apply the projection of u onto v according to the
   * <a href="https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process">Gram-Schmidt process</a>.
   *
   * @param u Vector on which v will be projected
   * @param v Vector which will be projected onto u
   * @return Projected vector
   */
  public static INDArray project(INDArray u, INDArray v) {
    // Check for correct lengths
    if (u.size(0) != v.size(0)) {
      throw new InvalidInputException(
	String.format("Size of u and v must be the same but is %d and %d",
	  u.size(0), v.size(0))
      );
    }

    final INDArray dotUV = t(u).mmul(v);
    final INDArray dotUU = t(u).mmul(u);
    final double div = dotUV.div(dotUU).getDouble(0);
    final INDArray proj = u.mul(div);
    return proj;
  }
}
