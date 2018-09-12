package nz.ac.waikato.cms.adams.multiway.data.tensor;

import nz.ac.waikato.cms.adams.multiway.exceptions.InvalidMethodCallException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.string.NDArrayStrings;

import java.util.Arrays;

/**
 * General multiway tensor for handling data.
 *
 * @author Steven Lang
 */
public class Tensor {

  /**
   * Number of decimals while printing data.
   */
  public static final int PRINT_PRECISION = 4;

  /** Underlying data. */
  protected INDArray data;

  /** Generate a tensor from an INDArray. */
  protected Tensor(INDArray data) {
    this.data = data.dup();
  }

  /**
   * Get the size of a certain dimension.
   *
   * @param dimension Dimension
   * @return Size of the given dimension
   */
  public long size(int dimension) {
    return data.size(dimension);
  }

  /**
   * Convert this Tensor into a 1D array. Throws a runtime exception if
   * underlying data is not 1D.
   *
   * @return 1D double representation of this tensor
   */
  public double toScalar() {
    final long[] shape = data.shape();
    // Assert correct order
    if (shape.length != 2 || (shape[0] != 1 || shape[1] != 1)) {
      throw new InvalidMethodCallException(String.format(
	"Method toArray1d was called on a %dD tensor with shape %s.",
        order(), Arrays.toString(shape)));
    }
    return this.data.getDouble(0, 0);
  }

  /**
   * Convert this Tensor into a 1D array. Throws a runtime exception if
   * underlying data is not 1D.
   *
   * @return 1D double representation of this tensor
   */
  public double[] toArray1d() {
    // Assert correct order
    final long[] shape = data.shape();
    if (shape.length != 2 || (shape[0] != 1 && shape[1] != 1)) {
      throw new InvalidMethodCallException(String.format(
	"Method toArray1d was called on a %dD tensor with shape %s.",
        order(), Arrays.toString(shape)));
    }

    int axis = shape[0] == 1 ? 1 : 0;

    final int dim0 = (int) data.size(axis);
    double[] arr = new double[dim0];
    for (int i = 0; i < dim0; i++) {
      arr[i] = data.getDouble(i);
    }
    return arr;
  }

  /**
   * Convert this Tensor into a 2D array. Throws a runtime exception if
   * underlying data is not 2D.
   *
   * @return 2D double representation of this tensor
   */
  public double[][] toArray2d() {
    // Assert correct order
    if (data.shape().length != 2) {
      throw new InvalidMethodCallException(String.format("Method toArray2d was called on a %dD tensor.", order()));
    }

    final int dim0 = (int) data.size(0);
    final int dim1 = (int) data.size(1);
    double[][] arr = new double[dim0][dim1];
    for (int i = 0; i < dim0; i++) {
      for (int j = 0; j < dim1; j++) {
	arr[i][j] = data.getDouble(i, j);
      }
    }
    return arr;
  }

  /**
   * Convert this Tensor into a 3D array. Throws a runtime exception if
   * underlying data is not 3D.
   *
   * @return 3D double representation of this tensor
   */
  public double[][][] toArray3d() {
    // Assert correct order
    if (data.shape().length != 3) {
      throw new InvalidMethodCallException(String.format("Method toArray3d was called on a %dD tensor.", order()));
    }

    final int dim0 = (int) data.size(0);
    final int dim1 = (int) data.size(1);
    final int dim2 = (int) data.size(2);
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
    return new Tensor(d);
  }

  /**
   * Create a tensor from 1D data.
   *
   * @param data 1D data
   * @return Tensor of the data
   */
  public static Tensor create(double[] data) {
    return new Tensor(Nd4j.create(data));
  }

  /**
   * Create a tensor from 2D data.
   *
   * @param data 2D data
   * @return Tensor of the data
   */
  public static Tensor create(double[][] data) {
    return new Tensor(Nd4j.create(data));
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
    return new Tensor(X);
  }

  /**
   * Create a tensor from an INDArray data.
   *
   * @param data INDArray
   * @return Tensor of the data
   */
  public static Tensor create(INDArray data) {
    return new Tensor(data);
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
    return Tensor.create(data.getRow(rowIndex));
  }

  /**
   * Get a set of rows.
   *
   * @param rowIndices Row indices
   * @return Subtensor of rows at the given indices
   */
  public Tensor getRows(int... rowIndices) {
    return Tensor.create(data.getRows(rowIndices));
  }

  /**
   * Get a specific column.
   *
   * @param columnIndex Column index
   * @return Column of this tensor at the given index
   */
  public Tensor getColumn(int columnIndex) {
    return Tensor.create(data.getColumn(columnIndex));
  }

  /**
   * Get a set of columns.
   *
   * @param columnIndices Column indices
   * @return Subtensor of columns at the given indices
   */
  public Tensor getColumns(int... columnIndices) {
    return Tensor.create(data.getColumns(columnIndices));
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
    return Tensor.create(data.transposei());
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
    return Tensor.create(data);
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
    if (!(obj instanceof Tensor)) {
      return false;
    }
    return data.equals(((Tensor) obj).getData());
  }

  public boolean equalsWithEps(Tensor other, double eps) {
    return data.equalsWithEps(other.getData(), eps);
  }
}
