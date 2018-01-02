package com.github.waikatodatamining.multiway.exceptions;

/**
 * Exception which indicates an invalid input.
 *
 * @author Steven Lang
 */
public class InvalidInputException extends RuntimeException {

  /** Serial version UID */
  private static final long serialVersionUID = 8824512314268137533L;

  public InvalidInputException() {}

  public InvalidInputException(String message) {
    super("Invalid input: " + message);
  }
}
