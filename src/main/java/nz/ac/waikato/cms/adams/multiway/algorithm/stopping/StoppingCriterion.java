package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

import lombok.extern.slf4j.Slf4j;

/**
 * Abstract class for iteration stopping criteria.
 *
 * @author Steven Lang
 */
@Slf4j
public abstract class StoppingCriterion<T> {

  /**
   * Check if the stopping criterion is met.
   *
   * @return True if the stopping criterion is met
   */
  public abstract boolean matches();


  /**
   * Update the value against which the criterion is checked.
   *
   * @param t Current value against which the criterion is checked
   */
  public void update(T t) {
    throw new UnsupportedOperationException(getType() + " criterion cannot be " +
      "updated with an external value.");
  }

  /**
   * Update the criterion's state.
   */
  public void update() {
    throw new UnsupportedOperationException(getType() +
      " criterion cannot be updated without an external value.");
  }

  /**
   * Get the criterion type.
   *
   * @return Criterion type
   */
  public abstract CriterionType getType();


  /**
   * Compare with another criterion and check if it is of the same type.
   *
   * @param other Other stoppping criterion
   * @return True if other stopping criterion is of the same type
   */
  public boolean sameTypeAs(StoppingCriterion other) {
    return getType().equals(other.getType());
  }

  /**
   * Reset the internal state.
   */
  public abstract void reset();

  /**
   * Check if criterion parameters are valid.
   */
  protected abstract void validateParameters();

  /**
   * Notify.
   *
   * @param msg Message
   */
  final void notify(String msg) {
    log.debug("{} stopping criterion was met: {}", getType(), msg);
  }
}
