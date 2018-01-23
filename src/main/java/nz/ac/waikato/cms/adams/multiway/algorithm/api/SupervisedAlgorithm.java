package nz.ac.waikato.cms.adams.multiway.algorithm.api;

import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

/**
 * Abstract class for supervised algorithms. Similar to
 * {@link UnsupervisedAlgorithm}, while all defined methods accept an additional
 * {@link Tensor} with dependent variables.
 *
 * @author Steven Lang
 */
public abstract class SupervisedAlgorithm extends AbstractAlgorithm {

  private static final long serialVersionUID = 6756079034435363940L;

  /**
   * Check the input and return an error message if something went wrong, else
   * null.
   *
   * @param x Data tensor
   * @param y Learning target
   * @return Error message if error, else null
   */
  protected abstract String check(Tensor x, Tensor y);

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
    String result = check(x, y);
    if (result == null)
      result = doBuild(x, y);
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
