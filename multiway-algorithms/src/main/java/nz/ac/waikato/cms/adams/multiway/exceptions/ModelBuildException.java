package nz.ac.waikato.cms.adams.multiway.exceptions;

/**
 * Exception which indicates an error while building the model.
 *
 * @author Steven Lang
 */
public class ModelBuildException extends RuntimeException {

  /** Serial version UID */
  private static final long serialVersionUID = 4593483282499651217L;

  public ModelBuildException() {
  }

  public ModelBuildException(String message) {
    super("Exception while building the model: " + message);
  }
}
