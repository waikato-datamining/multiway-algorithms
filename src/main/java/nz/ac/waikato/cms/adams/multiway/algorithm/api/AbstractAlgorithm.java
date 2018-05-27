package nz.ac.waikato.cms.adams.multiway.algorithm.api;

import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.Criterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.exceptions.UnsupportedStoppingCriterionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Abstract algorithm.
 *
 * @author Steven Lang
 */
public abstract class AbstractAlgorithm implements Serializable, Cloneable {

  private static Logger logger = LoggerFactory.getLogger(AbstractAlgorithm.class);

  /** Serial version UID */
  private static final long serialVersionUID = -4442219132263071797L;

  /** Stopping criteria */
  protected Map<CriterionType, Criterion> stoppingCriteria;

  /** Flag if the algorithm is finished */
  protected boolean isFinished;

  /** Enable debug mode */
  protected boolean isDebug;

  /** Default constructor. */
  public AbstractAlgorithm() {
    initialize();
    finishInitialize();
  }

  /** Initialize the internal algorithm state. */
  protected void initialize() {
    this.stoppingCriteria = new HashMap<>();
    this.isFinished = false;
  }

  /** Finish the internal algorithm state initialization. */
  protected void finishInitialize() {
    // No-op by default
  }

  /**
   * Get all stopping criteria.
   *
   * @return Array of stopping criteria
   */
  public Criterion[] getStoppingCriteria() {
    return stoppingCriteria.values().toArray(new Criterion[stoppingCriteria.size()]);
  }

  /**
   * Set all stopping criteria.
   *
   * @param stoppingCriteria Array of stopping criteria
   */
  public void setStoppingCriteria(Criterion[] stoppingCriteria) {
    for (Criterion c : stoppingCriteria) {
      this.stoppingCriteria.put(c.getType(), c);
    }
  }

  /**
   * Add a stopping criterion. If the same criterion type already exists, the old criterion will be
   * overwritten.
   *
   * @param criterion Stopping criterion
   */
  public void addStoppingCriterion(Criterion criterion) {
    if (!getAvailableStoppingCriteria().contains(criterion.getType())) {
      throw new UnsupportedStoppingCriterionException(
	"This algorithm does not" + "support " + criterion.getType() + " as stopping criterion.");
    }

    // Add new criterion
    stoppingCriteria.put(criterion.getType(), criterion);
  }

  /**
   * Check if any of the stopping criteria match.
   *
   * @return True of any of the stopping criteria match
   */
  protected boolean stoppingCriteriaMatch() {
    return stoppingCriteria.values().stream().anyMatch(Criterion::matches);
  }

  /** Get all available stopping criteria. */
  protected abstract Set<CriterionType> getAvailableStoppingCriteria();

  /** Reset stopping criteria states. */
  protected void resetStoppingCriteria() {
    stoppingCriteria.values().forEach(Criterion::reset);
  }

  /** Reset internal state. */
  protected void resetState() {
    this.isFinished = false;
    resetStoppingCriteria();
  }

  /** Return whether the algorithm is finished yet. */
  protected boolean isFinished() {
    return this.isFinished;
  }

  /**
   * Check if algorithm runs in debug mode.
   *
   * @return If algorithm runs in debug mode.
   */
  public boolean isDebug() {
    return isDebug;
  }

  /**
   * Set algorithm in debug mode.
   *
   * @param debug Debug flag.
   */
  public void setDebug(boolean debug) {
    isDebug = debug;
  }

  /**
   * Stops execution when the next iteration starts.
   */
  public void stopExecution() {
    logger.debug("Stop execution invoked. Algorithm will stop at next iteration.");
    this.stoppingCriteria.put(CriterionType.KILL, CriterionUtils.kill());
  }

  /**
   * Check if execution should be forcefully stopped.
   * @return True if {@link AbstractAlgorithm#stopExecution()} has been called.
   */
  protected boolean isForceStop(){
    return stoppingCriteria.containsKey(CriterionType.KILL);
  }

  //  /**
  //   * Preprocessing step. Return error message if something went wrong, else
  //   * null.
  //   *
  //   * @param input Input data
  //   * @return Error message if error, else null
  //   */
  //  protected  String preProcess(Tensor input) {
  //    return null;
  //  }
  //
  //  /**
  //   * Actual process implementation. Return error message if something went wrong, else
  //   * null.
  //   *
  //   * @param input Input data
  //   * @return Output data
  //   */
  //  protected abstract Tensor doProcess(Tensor input);
  //
  //  /**
  //   * Postprocessing step. Return error message if something went wrong, else
  //   * null.
  //   *
  //   * @param input  Input data
  //   * @param output Output data
  //   * @return Error message if error, else null
  //   */
  //  protected String postProcess(Tensor input, Tensor output) {
  //    return null;
  //  }
  //
  //  /**
  //   * Process method for the user.
  //   *
  //   * @param input Input data
  //   * @return Output data
  //   */
  //  public final Tensor process(Tensor input) {
  //    String msg = preProcess(input);
  //    if (msg != null)
  //      throw new IllegalStateException(msg);
  //    Tensor output = doProcess(input);
  //    msg = postProcess(input, output);
  //    if (msg != null)
  //      throw new IllegalStateException(msg);
  //    return output;
  //  }
}
