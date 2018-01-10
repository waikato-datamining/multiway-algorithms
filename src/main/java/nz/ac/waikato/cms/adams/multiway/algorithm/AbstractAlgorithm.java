package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.Criterion;
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
public abstract class AbstractAlgorithm implements Serializable{

  /** Serial version UID */
  private static final long serialVersionUID = -4442219132263071797L;

  /** Stopping criteria */
  protected List<Criterion> stoppingCriteria;

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
  boolean stoppingCriteriaMatch() {
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
}
