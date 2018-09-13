package nz.ac.waikato.cms.adams.multiway.algorithm.api;

import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Abstract class for multiblock supervised algorithms.
 *
 * @author Steven Lang
 */
public abstract class MultiBlockSupervisedAlgorithm extends AbstractAlgorithm {

  private static final Logger logger = LoggerFactory.getLogger(MultiBlockSupervisedAlgorithm.class);

  private static final long serialVersionUID = 4515752600074655745L;

  /**
   * Check the input and return an error message if something went wrong, else
   * null.
   *
   * @param x Data Blocks of Tensors
   * @param y Learning target
   * @return Error message if error, else null
   */
  protected String check(Tensor[] x, Tensor y){
    // Check for null
    if (x == null){
      return "Input data tensor must not be null.";
    }

    // Check blocks for null
    for (int i = 0; i < x.length; i++) {
      Tensor tensor = x[i];
      if (tensor == null) {
        return "Input data tensor at position " + i + " must not be null.";
      }
    }

    // Check target for null
    if (y == null){
      return "Input target tensor must not be null.";
    }


    return null;
  }

  /**
   * Run the actual build. Return error message if something went wrong, else
   * null.
   *
   * @param x Data Blocks of Tensors
   * @param y Learning target
   * @return Error message if error, else null
   */
  protected abstract String doBuild(Tensor[] x, Tensor y);

  /**
   * Public build method for the user. Return error message if something went
   * wrong, else null.
   *
   * @param x Data Blocks of Tensors
   * @param y Learning target
   * @return Error message if error, else null
   */
  public final String build(Tensor[] x, Tensor y) {
    String result = check(x, y);
    if (isDebug && result != null){
      logger.warn("Check(input) result was: {}", result);
    }
    if (result == null)
      result = doBuild(x, y);
    isFinished = true;
    return result;
  }

  /**
   * Make a new prediction of new datapoints.
   *
   * @param x New datapoints
   * @return Predictions of the new datapoints
   */
  public abstract Tensor predict(Tensor[] x);

}
