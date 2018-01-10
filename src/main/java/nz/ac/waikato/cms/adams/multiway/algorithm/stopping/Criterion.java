package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.Serializable;

/**
 * Abstract class for criteria.
 *
 * @author Steven Lang
 */
public abstract class Criterion<T> implements Serializable{

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(Criterion.class);

  /** Serial version UID */
  private static final long serialVersionUID = 6749729844847026611L;

  /**
   * Default constructor.
   */
  public Criterion() {
    initialize();
    finishInitialize();
  }


  /**
   * Initialize the internal criterion state.
   */
  protected void initialize() {
    // No-op by default
  }

  /**
   * Finish the internal criterion state initialization.
   */
  protected void finishInitialize() {
    // No-op by default
  }

  /**
   * Check if the criterion is met.
   *
   * @return True if the criterion is met
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
   * @param other Other criterion
   * @return True if other criterion is of the same type
   */
  public boolean sameTypeAs(Criterion other) {
    return getType().equals(other.getType());
  }

  /**
   * Reset the internal state.
   */
  public abstract void reset();

  /**
   * Notify.
   *
   * @param msg Message
   */
  protected void notify(String msg) {
    log.debug("{} criterion was met: {}", getType(), msg);
  }
}
