package com.github.waikatodatamining.multiway;

import java.lang.reflect.Field;

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
  public static double[][][] generateRandomTensor(int dim1, int dim2, int dim3) {
    double[][][] data= new double[dim1][dim2][dim3];
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        for (int k = 0; k < dim3; k++) {
          data[i][j][k] = Math.random();
        }
      }
    }
//    INDArray data = Nd4j.randn(new int[]{dim1, dim2, dim3}, 0);
    return data;
  }
}
