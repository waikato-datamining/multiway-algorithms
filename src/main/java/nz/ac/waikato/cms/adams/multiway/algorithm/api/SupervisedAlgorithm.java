package nz.ac.waikato.cms.adams.multiway.algorithm.api;

import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor.twoWayToThreeWay;

/**
 * Abstract class for supervised algorithms. Similar to
 * {@link UnsupervisedAlgorithm}, while all defined methods accept an additional
 * {@link Tensor} with dependent variables.
 *
 * @author Steven Lang
 */
public abstract class SupervisedAlgorithm extends AbstractAlgorithm {

  private static final long serialVersionUID = 6756079034435363940L;

  private static final Logger logger = LoggerFactory.getLogger(SupervisedAlgorithm.class);

  /**
   * Check the input and return an error message if something went wrong, else
   * null.
   *
   * @param x Data tensor
   * @param y Learning target
   * @return Error message if error, else null
   */
  protected String check(Tensor x, Tensor y){
    // Check for null
    if (x == null){
      return "Input data tensor must not be null.";
    }

    if (y == null){
      return "Input target tensor must not be null.";
    }


    return null;
  }

  /**
   * Run the actual build. Return error message if something went wrong, else
   * null.
   *
   * @param x Data tensor
   * @param y Learning target
   * @return Error message if error, else null
   */
  protected abstract String doBuild(Tensor x, Tensor y);

  /**
   * Public build method for the user. Return error message if something went
   * wrong, else null.
   *
   * @param x Data tensor
   * @param y Learning target
   * @return Error message if error, else null
   */
  public final String build(Tensor x, Tensor y) {

    // Check input
    String result = check(x, y);
    if (isDebug && result != null){
      logger.warn("Check(input) result was: {}", result);
    }


    if (result == null){
      // If input is two-way, transform to pseudo threeway ([10,5] -> [10,5,1])
      if (x.getData().rank() == 2) {
        x = twoWayToThreeWay(x);
      }
      result = doBuild(x, y);
    }
    isFinished = true;
    return result;
  }

  /**
   * Make a new prediction of new datapoints.
   *
   * @param x New datapoints
   * @return Predictions of the new datapoints
   */
  public abstract Tensor predict(Tensor x);

}
