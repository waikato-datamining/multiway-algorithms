package nz.ac.waikato.cms.adams.multiway.data.tensor;

import java.util.Map;

/**
 * General multiway tensor for handling data.
 *
 * @author Steven Lang
 */
public interface Tensor {

  /** Number of decimals while printing data. */
  int PRINT_PRECISION = 4;

  /**
   * Get the size of a certain dimension.
   *
   * @param dimension Dimension
   * @return Size of the given dimension
   */
  int size(int dimension);

  /**
   * Convert this Tensor into a 1D array. Throws a runtime exception if underlying data is not 1D.
   *
   * @return 1D double representation of this tensor
   */
  double toScalar();

  /**
   * Convert this Tensor into a 1D array. Throws a runtime exception if underlying data is not 1D.
   *
   * @return 1D double representation of this tensor
   */
  double[] toArray1d();

  /**
   * Convert this Tensor into a 2D array. Throws a runtime exception if underlying data is not 2D.
   *
   * @return 2D double representation of this tensor
   */
  double[][] toArray2d();

  /**
   * Convert this Tensor into a 3D array. Throws a runtime exception if underlying data is not 3D.
   *
   * @return 3D double representation of this tensor
   */
  double[][][] toArray3d();

  /**
   * Order of the tensor.
   *
   * @return Order of the tensor
   */
  int order();

  /**
   * Get a specific double value from the indices.
   *
   * @param indices Indices
   * @return Double value at {@code indices}
   */
  double getDouble(int... indices);

  /**
   * Get a specific row.
   *
   * @param rowIndex Row index
   * @return Row of this tensor at the given index
   */
  Tensor getRow(int rowIndex);

  /**
   * Get a set of rows.
   *
   * @param rowIndices Row indices
   * @return Subtensor of rows at the given indices
   */
  Tensor getRows(int... rowIndices);

  /**
   * Get a specific column.
   *
   * @param columnIndex Column index
   * @return Column of this tensor at the given index
   */
  Tensor getColumn(int columnIndex);

  /**
   * Get a set of columns.
   *
   * @param columnIndices Column indices
   * @return Subtensor of columns at the given indices
   */
  Tensor getColumns(int... columnIndices);

  /**
   * Transpose the tensor.
   *
   * @return Transposed tensor
   */
  Tensor t();

  /**
   * Transpose the tensor inplace.
   *
   * @return Transposed tensor
   */
  Tensor ti();

  /**
   * Duplicate this tensor.
   *
   * @return Duplicate of this tensor
   */
  Tensor dup();

  Tensor mean(int... dimensions);

  Tensor std(int... dimensions);

  Tensor divRowVector(Tensor row);

  Tensor divColumnVector(Tensor column);

  Tensor mulRowVector(Tensor row);

  Tensor mulColumnVector(Tensor column);

  Tensor addRowVector(Tensor row);

  Tensor addColumnVector(Tensor column);

  Tensor subRowVector(Tensor row);

  Tensor subColumnVector(Tensor column);

  Tensor mmul(Tensor other);

  Tensor div(Tensor other);

  Tensor norm2(int... dimensions);

  Tensor add(Tensor other);

  Tensor sub(Tensor other);

  Tensor div(double val);

  Tensor mul(double val);

  int columns();

  int rows();

  Tensor invert();

  Tensor invert(boolean inPlace);

  Tensor reshape(int... newShape);

  Tensor normalize();

  double squaredDistance(Tensor other);

  Tensor broadcast(int... shape);

  int[] shape();

  Tensor sum(int... dimension);

  void putColumn(int column, Tensor data);

  void putRow(int row, Tensor data);

  public Map<String, Tensor> svd(Tensor x);

  Tensor concat(Tensor other, int axis);

  Tensor permute(int... permutation);
}
