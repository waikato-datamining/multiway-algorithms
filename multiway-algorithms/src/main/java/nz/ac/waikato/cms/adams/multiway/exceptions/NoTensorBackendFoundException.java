package nz.ac.waikato.cms.adams.multiway.exceptions;

/**
 * Exception which is thrown when no Tensor implementation could be found.
 *
 * @author Steven Lang
 */
public class NoTensorBackendFoundException extends RuntimeException {

  /** Serial version UID */
  private static final long serialVersionUID = 8824512314268137533L;

  public NoTensorBackendFoundException() {}

  public NoTensorBackendFoundException(String message) {
    super("No Tensor implementation could be found. " + message);
  }
}
