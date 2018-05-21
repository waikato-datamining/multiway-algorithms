package data.tensor;

import com.google.common.collect.ImmutableMap;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.exceptions.InvalidMethodCallException;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.string.NDArrayStrings;

import java.util.Arrays;
import java.util.Map;

/**
 * General multiway tensor for handling data.
 *
 * @author Steven Lang
 */
public class Nd4jTensor implements Tensor {

  /** Number of decimals while printing data. */
  public static int PRINT_PRECISION = 4;

  /** Underlying data. */
  protected INDArray data;

  /** Generate a tensor from an INDArray. */
  protected Nd4jTensor(INDArray data) {
    this.data = data.dup();
  }

  /**
   * Get the size of a certain dimension.
   *
   * @param dimension Dimension
   * @return Size of the given dimension
   */
  public int size(int dimension) {
    return data.size(dimension);
  }

  /**
   * Convert this Tensor into a 1D array. Throws a runtime exception if underlying data is not 1D.
   *
   * @return 1D double representation of this tensor
   */
  public double toScalar() {
    final int[] shape = data.shape();
    // Assert correct order
    if (shape.length != 2 || (shape[0] != 1 || shape[1] != 1)) {
      throw new InvalidMethodCallException(
          String.format(
              "Method toArray1d was called on a %dD tensor with shape %s.",
              order(), Arrays.toString(shape)));
    }
    return this.data.getDouble(0, 0);
  }

  /**
   * Convert this Tensor into a 1D array. Throws a runtime exception if underlying data is not 1D.
   *
   * @return 1D double representation of this tensor
   */
  public double[] toArray1d() {
    // Assert correct order
    final int[] shape = data.shape();
    if (shape.length != 2 || (shape[0] != 1 && shape[1] != 1)) {
      throw new InvalidMethodCallException(
          String.format(
              "Method toArray1d was called on a %dD tensor with shape %s.",
              order(), Arrays.toString(shape)));
    }

    int axis = shape[0] == 1 ? 1 : 0;

    final int dim0 = data.size(axis);
    double[] arr = new double[dim0];
    for (int i = 0; i < dim0; i++) {
      arr[i] = data.getDouble(i);
    }
    return arr;
  }

  /**
   * Convert this Tensor into a 2D array. Throws a runtime exception if underlying data is not 2D.
   *
   * @return 2D double representation of this tensor
   */
  public double[][] toArray2d() {
    // Assert correct order
    if (data.shape().length != 2) {
      throw new InvalidMethodCallException(
          String.format("Method toArray2d was called on a %dD tensor.", order()));
    }

    final int dim0 = data.size(0);
    final int dim1 = data.size(1);
    double[][] arr = new double[dim0][dim1];
    for (int i = 0; i < dim0; i++) {
      for (int j = 0; j < dim1; j++) {
        arr[i][j] = data.getDouble(i, j);
      }
    }
    return arr;
  }

  /**
   * Convert this Tensor into a 3D array. Throws a runtime exception if underlying data is not 3D.
   *
   * @return 3D double representation of this tensor
   */
  public double[][][] toArray3d() {
    // Assert correct order
    if (data.shape().length != 3) {
      throw new InvalidMethodCallException(
          String.format("Method toArray3d was called on a %dD tensor.", order()));
    }

    final int dim0 = data.size(0);
    final int dim1 = data.size(1);
    final int dim2 = data.size(2);
    double[][][] arr = new double[dim0][dim1][dim2];
    for (int i = 0; i < dim0; i++) {
      for (int j = 0; j < dim1; j++) {
        for (int k = 0; k < dim2; k++) {
          arr[i][j][k] = data.getDouble(i, j, k);
        }
      }
    }
    return arr;
  }

  /**
   * Order of the tensor.
   *
   * @return Order of the tensor
   */
  public int order() {
    return data.shape().length;
  }

  /**
   * Create a tensor from a scalar.
   *
   * @param data Scalar data
   * @return Tensor of the data
   */
  public static Tensor create(double data) {
    final INDArray d = Nd4j.create(1, 1);
    d.putScalar(0, 0, data);
    return new Nd4jTensor(d);
  }

  /**
   * Create a tensor from 1D data.
   *
   * @param data 1D data
   * @return Tensor of the data
   */
  public static Tensor create(double[] data) {
    return new Nd4jTensor(Nd4j.create(data));
  }

  /**
   * Create a tensor from 2D data.
   *
   * @param data 2D data
   * @return Tensor of the data
   */
  public static Tensor create(double[][] data) {
    return new Nd4jTensor(Nd4j.create(data));
  }

  /**
   * Create a tensor from 3D data.
   *
   * @param data 3D data
   * @return Tensor of the data
   */
  public static Tensor create(double[][][] data) {
    int numRows = data.length;
    int numColumns = data[0].length;
    int numDimensions = data[0][0].length;
    // Create array of slices
    INDArray X = Nd4j.create(numRows, numColumns, numDimensions);
    for (int i = 0; i < data.length; i++) {
      double[][] slice = data[i];
      X.putRow(i, Nd4j.create(slice));
    }
    return new Nd4jTensor(X);
  }

  /**
   * Create a tensor from an INDArray data.
   *
   * @param data INDArray
   * @return Tensor of the data
   */
  public static Tensor create(INDArray data) {
    return new Nd4jTensor(data);
  }

  /**
   * Get a specific double value from the indices.
   *
   * @param indices Indices
   * @return Double value at {@code indices}
   */
  public double getDouble(int... indices) {
    return data.getDouble(indices);
  }

  /**
   * Get a specific row.
   *
   * @param rowIndex Row index
   * @return Row of this tensor at the given index
   */
  public Tensor getRow(int rowIndex) {
    return Nd4jTensor.create(data.getRow(rowIndex));
  }

  /**
   * Get a set of rows.
   *
   * @param rowIndices Row indices
   * @return Subtensor of rows at the given indices
   */
  public Tensor getRows(int... rowIndices) {
    return Nd4jTensor.create(data.getRows(rowIndices));
  }

  /**
   * Get a specific column.
   *
   * @param columnIndex Column index
   * @return Column of this tensor at the given index
   */
  public Tensor getColumn(int columnIndex) {
    return Nd4jTensor.create(data.getColumn(columnIndex));
  }

  /**
   * Get a set of columns.
   *
   * @param columnIndices Column indices
   * @return Subtensor of columns at the given indices
   */
  public Tensor getColumns(int... columnIndices) {
    return Nd4jTensor.create(data.getColumns(columnIndices));
  }

  /**
   * Get the INDArray representing the data beneath.
   *
   * @return INDArray of the data
   */
  public INDArray getData() {
    return data;
  }

  /**
   * Transpose the tensor.
   *
   * @return Transposed tensor
   */
  public Tensor t() {
    return Nd4jTensor.create(data.transposei());
  }

  /**
   * Transpose the tensor inplace.
   *
   * @return Transposed tensor
   */
  public Tensor ti() {
    data.transposei();
    return this;
  }

  /**
   * Duplicate this tensor.
   *
   * @return Duplicate of this tensor
   */
  public Tensor dup() {
    return Nd4jTensor.create(data);
  }

  public Tensor mean(int... dimensions) {
    return Nd4jTensor.create(data.mean(dimensions));
  }

  public Tensor std(int... dimensions) {
    return Nd4jTensor.create(data.std(dimensions));
  }

  public Tensor divRowVector(Tensor row) {
    final INDArray data = ((Nd4jTensor) row).getData();
    return Nd4jTensor.create(this.data.divRowVector(data));
  }

  public Tensor divColumnVector(Tensor column) {
    final INDArray data = ((Nd4jTensor) column).getData();
    return Nd4jTensor.create(this.data.divColumnVector(data));
  }

  public Tensor mulRowVector(Tensor row) {
    final INDArray data = ((Nd4jTensor) row).getData();
    return Nd4jTensor.create(this.data.mulRowVector(data));
  }

  public Tensor mulColumnVector(Tensor column) {
    final INDArray data = ((Nd4jTensor) column).getData();
    return Nd4jTensor.create(this.data.mulColumnVector(data));
  }

  public Tensor addRowVector(Tensor row) {
    final INDArray data = ((Nd4jTensor) row).getData();
    return Nd4jTensor.create(this.data.addRowVector(data));
  }

  public Tensor addColumnVector(Tensor column) {
    final INDArray data = ((Nd4jTensor) column).getData();
    return Nd4jTensor.create(this.data.addColumnVector(data));
  }

  public Tensor subRowVector(Tensor row) {
    final INDArray data = ((Nd4jTensor) row).getData();
    return Nd4jTensor.create(this.data.subRowVector(data));
  }

  public Tensor subColumnVector(Tensor column) {
    final INDArray data = ((Nd4jTensor) column).getData();
    return Nd4jTensor.create(this.data.subColumnVector(data));
  }

  public Tensor mmul(Tensor other) {
    final INDArray data = ((Nd4jTensor) other).getData();
    return Nd4jTensor.create(this.data.mmul(data));
  }

  public Tensor norm2(int... dimensions) {
    return Nd4jTensor.create(this.data.norm2(dimensions));
  }

  public Tensor add(Tensor other) {
    final INDArray data = ((Nd4jTensor) other).getData();
    return Nd4jTensor.create(this.data.add(data));
  }

  public Tensor sub(Tensor other) {
    final INDArray data = ((Nd4jTensor) other).getData();
    return Nd4jTensor.create(this.data.sub(data));
  }

  public Tensor div(double val) {
    return Nd4jTensor.create(this.data.div(val));
  }

  public int columns() {
    return data.columns();
  }

  public int rows() {
    return data.rows();
  }

  public Tensor invert() {
    return invert(false);
  }

  public Tensor reshape(int... newShape) {
    return Nd4jTensor.create(this.data.reshape(newShape));
  }

  public Tensor normalize() {
    return Nd4jTensor.create(Transforms.unitVec(this.data));
  }

  /**
   * Build the inverse of a matrix
   *
   * @return the inverted matrix
   */
  public Tensor invert(boolean inPlace) {
    if (data.columns() != data.rows()) {
      throw new IllegalArgumentException("invalid array: must be square matrix");
    }
    RealMatrix rm = CheckUtil.convertToApacheMatrix(data);
    RealMatrix rmInverse = new LUDecomposition(rm).getSolver().getInverse();

    INDArray inverse = CheckUtil.convertFromApacheMatrix(rmInverse);
    if (inPlace) data.assign(inverse);
    return Nd4jTensor.create(inverse);
  }

  /**
   * Perform an SVD on the given matrix.
   *
   * @param x Input matrix
   * @return SVD matrices U,S,V
   */
  public Map<String, Tensor> svd(Tensor x) {
    INDArray data = ((Nd4jTensor) x).getData();
    final RealMatrix xApache = CheckUtil.convertToApacheMatrix(data);
    SingularValueDecomposition svd = new SingularValueDecomposition(xApache);
    final double[] singularValues = svd.getSingularValues();
    final INDArray U = CheckUtil.convertFromApacheMatrix(svd.getU());
    final INDArray S = CheckUtil.convertFromApacheMatrix(svd.getS());
    final INDArray V = CheckUtil.convertFromApacheMatrix(svd.getV());
    final INDArray SVAL = Nd4j.create(singularValues).transpose();
    return ImmutableMap.of(
        "U", Nd4jTensor.create(U),
        "S", Nd4jTensor.create(S),
        "V", Nd4jTensor.create(V),
        "SVAL", Nd4jTensor.create(SVAL));
  }

  public double squaredDistance(Tensor other) {
    final INDArray data = ((Nd4jTensor) other).getData();
    return this.data.squaredDistance(data);
  }

  /**
   * Concatenate a matrix and a vector at a given axis. Still valid if {@code this.data} is {@code
   * null}, as {@code this.data} will then be initialized with {@code other} as first vector at the
   * given axis {@code axis}.
   *
   * @param other Vector
   * @param axis Concatenation axis
   * @return Concatenation of the matrix and the vector at the given axis
   */
  public Tensor concat(Tensor other, int axis) {
    final INDArray uaReshaped = ((Nd4jTensor) other).getData().reshape(-1, 1);
    if (this.data == null) {
      return Nd4jTensor.create(uaReshaped);
    } else {
      return Nd4jTensor.create(Nd4j.concat(axis, this.data, uaReshaped));
    }
  }

  public Tensor div(Tensor other) {
    INDArray data = ((Nd4jTensor)other).getData();
    return Nd4jTensor.create(this.data.div(data));
  }

  public Tensor mul(double val) {
    return Nd4jTensor.create(this.data.mul(val));
  }

  public Tensor broadcast(int... shape) {
    return Nd4jTensor.create(this.data.broadcast(shape));
  }

  public int[] shape() {
    return this.data.shape();
  }

  public Tensor sum(int... dimensions) {
    return Nd4jTensor.create(this.data.sum(dimensions));
  }

  public void putColumn(int column, Tensor data) {
    final INDArray columnData = ((Nd4jTensor) data).getData();
    this.data.putColumn(column, columnData);
  }

  public void putRow(int row, Tensor data) {
    final INDArray rowData = ((Nd4jTensor) data).getData();
    this.data.putColumn(row, rowData);
  }

  public Tensor permute(int... permutation) {
    return Nd4jTensor.create(this.data.permute(permutation));
  }

  @Override
  public String toString() {
    NDArrayStrings s = new NDArrayStrings(PRINT_PRECISION);
    return s.format(data);
  }

  @Override
  public int hashCode() {
    return data.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof Nd4jTensor)) {
      return false;
    }
    return data.equals(((Nd4jTensor) obj).getData());
  }
}
