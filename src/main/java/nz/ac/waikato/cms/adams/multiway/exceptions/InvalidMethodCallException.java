package nz.ac.waikato.cms.adams.multiway.exceptions;

/**
 * Exception which indicates an invalid method was called.
 *
 * @author Steven Lang
 */
public class InvalidMethodCallException extends RuntimeException {

  /** Serial version UID */
  private static final long serialVersionUID = 7966409363159598002L;

  public InvalidMethodCallException() {}

  public InvalidMethodCallException(String message) {
    super("Invalid method call: " + message);
  }
}
