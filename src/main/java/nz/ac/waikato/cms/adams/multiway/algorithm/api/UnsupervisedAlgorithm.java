package nz.ac.waikato.cms.adams.multiway.algorithm.api;

import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

public abstract class UnsupervisedAlgorithm extends AbstractAlgorithm {

  private static final long serialVersionUID = 6756079034435363940L;

  /**
   * Check the input and return an error message if something went wrong, else
   * null.
   *
   * @param x Data tensor
   * @return Error message if error, else null
   */
  protected abstract String check(Tensor x);

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
    if (result == null)
      result = doBuild(x);
    return result;
  }

}
