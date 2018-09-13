package nz.ac.waikato.cms.adams.multiway;

import nz.ac.waikato.cms.adams.multiway.data.DataReader;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Assert;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Random;

/**
 * Test utilities.
 *
 * @author Steven Lang
 */
public class TestUtils {

  /**
   * Get a private field of a certain object by name.
   *
   * @param obj  Object to be accessed
   * @param name Fieldname
   * @param <T>  Class to return
   * @return Field value
   * @throws IllegalAccessException Access not allowed
   * @throws NoSuchFieldException   Field not found
   */
  public static <T> T getField(Object obj, String name) throws IllegalAccessException, NoSuchFieldException {
    Field field = obj.getClass().getDeclaredField(name);
    field.setAccessible(true);
    return (T) field.get(obj);
  }

  /**
   * Generate a random 3d data tensor
   *
   * @param dim1 First dimension
   * @param dim2 Second dimension
   * @param dim3 Third dimension
   * @return 3D Data tensor
   */
  public static Tensor generateRandomTensor(int dim1, int dim2, int dim3) {
    Random rng = new Random(0);
    double[][][] data = new double[dim1][dim2][dim3];
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
	for (int k = 0; k < dim3; k++) {
	  data[i][j][k] = rng.nextDouble() * 100;
	}
      }
    }
    return Tensor.create(data);
  }


  /**
   * Generate a tensor based on the given shape with increasing values for each
   * index.
   *
   * @param shape Tensor shape
   * @return Tensor
   */
  public static Tensor generateRangeTensor(int[] shape) {
    int range = Arrays.stream(shape).reduce(1, (l, r) -> l * r);
    return Tensor.create(Nd4j.arange(range).reshape(shape));
  }


  /**
   * Generate a random matrix tensor
   *
   * @param dim1 First dimension
   * @param dim2 Second dimension
   * @return Matrix
   */
  public static Tensor generateRandomMatrix(int dim1, int dim2) {
    Random rng = new Random(0);
    double[][] data = new double[dim1][dim2];
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
	data[i][j] = rng.nextDouble() * 100;
      }
    }
    return Tensor.create(data);
  }

  /**
   * Load the data.
   *
   * @return Data tensor
   */
  public static Tensor loadRegressionTestData() {
    String prefix = "src/test/resources/data/regression/unsupervised" +
      "/fluorescence/data";
    String suffix = ".csv";
    int startIdx = 1;
    int endIdx = 121;

    final double[][][] data;
    try {
      data = DataReader.read3WayMultiCsv(prefix, suffix, startIdx, endIdx, ",", false);
    }
    catch (IOException e) {
      e.printStackTrace();
      Assert.fail(e.getMessage());
      return null;
    }
    final Tensor tensor = Tensor.create(data);
    return tensor;
  }


  /**
   * Load the synthetic supervised dataset.
   *
   * @return Tensor array. T[0] = X, T[1] = Y
   */
  public static Tensor[] loadSyntheticSupervisedData() {
    String prefix = "src/test/resources/data/regression/supervised/synthetic/";
    try {
      double[][][] X = DataReader.read3WaySparse(prefix + "X-threeway.csv", " ", 3, false);
      double[][] Y = DataReader.readSparseMatrix(prefix + "Y-multitarget.csv", " ", false);
      return new Tensor[]{Tensor.create(X), Tensor.create(Y)};
    }
    catch (IOException e) {
      e.printStackTrace();
      Assert.fail(e.getMessage());
      return null;
    }
  }

  /**
   * Load the synthetic supervised dataset.
   *
   * @return Tensor array. T[0] = X1, T[1] = X2, T[2] = Y
   */
  public static Tensor[] loadSyntheticMultiBlockSupervisedData() {
    Tensor[] data = loadSyntheticSupervisedData();
    Tensor x1 = data[0];
    long[] x1Shape = x1.getData().shape();
    int[] x1IntShape = {(int) x1Shape[0], (int) x1Shape[1], (int) x1Shape[2]};
    INDArray randnLikeX = Nd4j.randn(x1IntShape, 0);
    Tensor x2 = Tensor.create(x1.getData().add(randnLikeX));
    Tensor y = data[1];

    return new Tensor[]{
      x1, x2, y
    };
  }
}
