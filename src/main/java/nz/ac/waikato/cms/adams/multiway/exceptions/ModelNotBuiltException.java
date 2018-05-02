package nz.ac.waikato.cms.adams.multiway.exceptions;

/**
 * Exception which indicates that a model method using the models internal state
 * has been invoked before the model has actually been built.
 *
 * @author Steven Lang
 */
public class ModelNotBuiltException extends RuntimeException {

  /** Serial version UID */
  private static final long serialVersionUID = 4061212241955898227L;


  public ModelNotBuiltException() {
  }

  public ModelNotBuiltException(String message) {
    super("The model has not been built. Message: " + message);
  }
}
