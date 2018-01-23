package nz.ac.waikato.cms.adams.multiway.algorithm.api;

import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.Criterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.exceptions.UnsupportedStoppingCriterionException;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Abstract algorithm.
 *
 * @author Steven Lang
 */
public abstract class AbstractAlgorithm implements Serializable {

  /** Serial version UID */
  private static final long serialVersionUID = -4442219132263071797L;

  /** Stopping criteria */
  protected List<Criterion> stoppingCriteria;

  /** Flag if the algorithm is finished */
  protected boolean isFinished;

  /**
   * Default constructor.
   */
  public AbstractAlgorithm() {
    initialize();
    finishInitialize();
  }

  /**
   * Initialize the internal algorithm state.
   */
  protected void initialize() {
    this.stoppingCriteria = new ArrayList<>();
    this.isFinished = false;
  }

  /**
   * Finish the internal algorithm state initialization.
   */
  protected void finishInitialize() {
    // No-op by default
  }

  /**
   * Get all stopping criteria.
   *
   * @return Array of stopping criteria
   */
  public Criterion[] getStoppingCriteria() {
    return stoppingCriteria.toArray(
      new Criterion[stoppingCriteria.size()]);
  }

  /**
   * Set all stopping criteria.
   *
   * @param stoppingCriteria Array of stopping criteria
   */
  public void setStoppingCriteria(Criterion[] stoppingCriteria) {
    this.stoppingCriteria = Arrays.stream(stoppingCriteria)
      .collect(Collectors.toList());
  }

  /**
   * Add a stopping criterion. If the same criterion type already exists, the
   * old criterion will be overwritten.
   *
   * @param criterion Stopping criterion
   */
  public void addStoppingCriterion(Criterion criterion) {
    if (!getAvailableStoppingCriteria().contains(criterion.getType())) {
      throw new UnsupportedStoppingCriterionException("This algorithm does not" +
	"support " + criterion.getType() + " as stopping criterion.");
    }

    // Remove same criterion if present
    stoppingCriteria.removeIf(c -> c.sameTypeAs(criterion));

    // Add new criterion
    stoppingCriteria.add(criterion);
  }

  /**
   * Check if any of the stopping criteria match.
   *
   * @return True of any of the stopping criteria match
   */
  protected boolean stoppingCriteriaMatch() {
    return stoppingCriteria.stream().anyMatch(Criterion::matches);
  }

  /**
   * Get all available stopping criteria.
   */
  protected abstract Set<CriterionType> getAvailableStoppingCriteria();

  /**
   * Reset stopping criteria states.
   */
  protected void resetStoppingCriteria() {
    stoppingCriteria.forEach(Criterion::reset);
  }

  /**
   * Reset internal state.
   */
  protected void resetState(){
    this.isFinished = false;
    resetStoppingCriteria();
  }

  /**
   * Return whether the algorithm is finished yet.
   */
  protected boolean isFinished() {
    return this.isFinished;
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
