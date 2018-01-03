package com.github.waikatodatamining.multiway.exceptions;

/**
 * Exception which indicates a stopping criterion was not supported.
 *
 * @author Steven Lang
 */
public class UnsupportedStoppingCriterionException extends RuntimeException {

  /** Serial version UID */
  public UnsupportedStoppingCriterionException() {}

  private static final long serialVersionUID = -6494795365713661302L;

  public UnsupportedStoppingCriterionException(String message) {
    super("Unsupported stopping criterion: " + message);
  }
}
