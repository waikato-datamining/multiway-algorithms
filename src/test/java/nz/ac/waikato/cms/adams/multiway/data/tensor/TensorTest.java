package nz.ac.waikato.cms.adams.multiway.data.tensor;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Testcases for {@link Tensor}.
 *
 * @author Steven Lang
 */
public class TensorTest {

  @Test
  public void toScalar() {
    double data = 0d;
    final Tensor tensor = Tensor.create(data);
    assertEquals(data, tensor.toScalar(), 10E-8);
  }

  @Test
  public void toArray1d() {
    double[] data = new double[10];
    final Tensor tensor = Tensor.create(data);
    assertArrayEquals(data, tensor.toArray1d(), 10E-8);
  }

  @Test
  public void toArray2d() {
    double[][] data = new double[10][10];
    final Tensor tensor = Tensor.create(data);
    final double[][] actual = tensor.toArray2d();
    for (int i = 0; i < 10; i++) {
      assertArrayEquals(data[i], actual[i], 10E-8);
    }
  }

  @Test
  public void toArray3d() {
    double[][][] data = new double[10][10][10];
    final Tensor tensor = Tensor.create(data);
    final double[][][] actual = tensor.toArray3d();
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        assertArrayEquals(data[i][j], actual[i][j], 10E-8);
      }
    }
  }
}