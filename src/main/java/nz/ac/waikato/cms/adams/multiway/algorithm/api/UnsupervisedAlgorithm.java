package nz.ac.waikato.cms.adams.multiway.algorithm.api;

import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class UnsupervisedAlgorithm extends AbstractAlgorithm {

  private static final long serialVersionUID = 6756079034435363940L;


  private static final Logger logger = LoggerFactory.getLogger(UnsupervisedAlgorithm.class);


  /**
   * Check the input and return an error message if something went wrong, else
   * null.
   *
   * @param x Data tensor
   * @return Error message if error, else null
   */
  protected String check(Tensor x){
    // Check for null
    if (x == null){
      return "Input tensor must not be null.";
    }
    return null;
  }

  /**
   * Run the actual build. Return error message if something went wrong, else
   * null.
   *
   * @param x Data tensor
   * @return Error message if error, else null
   */
  protected abstract String doBuild(Tensor x);

  /**
   * Public build method for the user. Return error message if something went
   * wrong, else null.
   *
   * @param x Data tensor
   * @return Error message if error, else null
   */
  public final String build(Tensor x) {
    String result = check(x);
    if (isDebug && result != null){
      logger.warn("Check(input) result was: {}", result);
    }
    if (result == null)
      result = doBuild(x);
    isFinished = true;
    return result;
  }

}
