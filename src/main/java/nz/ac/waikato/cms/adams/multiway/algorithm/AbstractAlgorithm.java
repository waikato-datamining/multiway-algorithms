package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.StoppingCriterion;
import nz.ac.waikato.cms.adams.multiway.exceptions.UnsupportedStoppingCriterionException;

import java.util.HashSet;
import java.util.Set;

/**
 * Abstract algorithm.
 *
 * @author Steven Lang
 */
public abstract class AbstractAlgorithm {

  /** Stopping criteria */
  protected Set<StoppingCriterion> stoppingCriteria;

  /**
   * Default constructor.
   */
  public AbstractAlgorithm() {
    this.stoppingCriteria = new HashSet<>();
  }

  /**
   * Add a stopping criterion. If the same criterion type already exists, the
   * old criterion will be overwritten.
   *
   * @param criterion Stopping criterion
   */
  public void addStoppingCriterion(StoppingCriterion criterion) {
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
  boolean stoppingCriteriaMatch() {
    return stoppingCriteria.stream().anyMatch(StoppingCriterion::matches);
  }

  /**
   * Update the internal state.
   */
  protected abstract void update();

  /**
   * Get all available stopping criteria.
   */
  protected abstract Set<CriterionType> getAvailableStoppingCriteria();

  /**
   * Reset stopping criteria states.
   */
  protected void resetStoppingCriteria(){
    stoppingCriteria.forEach(StoppingCriterion::reset);
  }
}
