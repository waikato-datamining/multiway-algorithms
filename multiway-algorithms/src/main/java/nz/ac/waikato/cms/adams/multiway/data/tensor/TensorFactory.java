package nz.ac.waikato.cms.adams.multiway.data.tensor;

import nz.ac.waikato.cms.adams.multiway.exceptions.NoTensorBackendFoundException;
import nz.ac.waikato.cms.locator.ClassLister;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

public class TensorFactory {

  private static TensorArrayFactory INSTANCE;
  private static Logger logger = LoggerFactory.getLogger(TensorFactory.class);

  static void initialize() {
    if (INSTANCE != null) {
      return;
    }

    // configuring the class hierarchies
    Properties pkgs = new Properties();
    pkgs.put(TensorFactory.class.getName(), "nz.ac.waikato.cms.adams.multiway.data.tensor");

    // initialize
    ClassLister lister = ClassLister.getSingleton();
    lister.setPackages(pkgs);
    lister.initialize();

    // interface
    System.out.println("\nBackends for: " + TensorArrayFactory.class.getName());
    Class[] classes = lister.getClasses(Tensor.class);
    for (Class cls : classes) {
      System.out.println("- " + cls.getName());
    }

    if (classes.length == 0) {
      throw new NoTensorBackendFoundException();
    } else if (classes.length > 1) {
      final String firstClassName = classes[0].getName();
      logger.warn(
          "Multiple Tensor implementations have been found. Using {} by default.", firstClassName);
    } else {
      try {
        INSTANCE = (TensorArrayFactory) classes[0].newInstance();
      } catch (InstantiationException | IllegalAccessException e) {
        e.printStackTrace();
      }
    }
  }

  public static Tensor zeros(int... shape) {
    initialize();
    return INSTANCE.zeros(shape);
  }

  public static Tensor ones(int... shape) {
    initialize();
    return INSTANCE.ones(shape);
  }

  public static Tensor randn(int rows, int columns, long seed) {
    initialize();
    return INSTANCE.randn(rows, columns, seed);
  }

  public static Tensor randn(int dim1, int dim2, int dim3, long seed) {
    initialize();
    return INSTANCE.randn(dim1, dim2, dim3, seed);
  }

  public static Tensor randn(int[] shape, long seed) {
    initialize();
    return INSTANCE.randn(shape, seed);
  }
}
