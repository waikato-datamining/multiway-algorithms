package com.github.waikatodatamining.multiway.algorithm.stopping;

/**
 * Interface for stopping criteria.
 *
 * @author Steven Lang
 */
public interface StoppingCriterion<T> {

  /**
   * Check if the stopping criterion is met.
   *
   * @return True if the stopping criterion is met
   */
  boolean matches();

  /**
   * Update the value against which the criterion is checked.
   *
   * @param t Current value against which the criterion is checked
   */
  default void update(T t) {
    throw new UnsupportedOperationException(getType() + " criterion cannot be " +
      "updated with an external value.");
  }

  /**
   * Update the criterion's state.
   */
  default void update() {
    throw new UnsupportedOperationException(getType() +
      " criterion cannot be updated without an external value.");
  }

  /**
   * Get the criterion type.
   *
   * @return Criterion type
   */
  CriterionType getType();


  /**
   * Compare with another criterion and check if it is of the same type.
   *
   * @param other Other stoppping criterion
   * @return True if other stopping criterion is of the same type
   */
  default boolean sameTypeAs(StoppingCriterion other) {
    return getType().equals(other.getType());
  }

  /**
   * Reset the internal state.
   */
  void reset();

  /**
   * Check if criterion parameters are valid.
   */
  void validateParameters();
}
