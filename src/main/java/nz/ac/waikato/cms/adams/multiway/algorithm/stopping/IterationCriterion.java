package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Criterion that checks if a maximum number of iterations is reached.
 *
 * @author Steven Lang
 */
public class IterationCriterion extends Criterion<Integer> {

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(IterationCriterion.class);


  /** Serial version UID */
  private static final long serialVersionUID = -4166075237246319251L;

  /** Current iteration */
  protected int currentIteration;

  /** Maximum number of iterations */
  protected int maxIterations;

  /**
   * Get maximum number of iterations.
   *
   * @return Maximum number of iterations
   */
  public int getMaxIterations() {
    return maxIterations;
  }

  /**
   * Set number of maximum iterations
   *
   * @param maxIterations Maximum iterations
   */
  public void setMaxIterations(int maxIterations) {
    if (maxIterations <= 0) {
      log.warn("Maximum number of iterations must be greater" +
	" than zero.");
    }
    else {
      this.maxIterations = maxIterations;
    }
  }

  @Override
  protected void initialize() {
    super.initialize();
    this.currentIteration = 0;
  }

  @Override
  public boolean matches() {
    final boolean m = currentIteration >= maxIterations;
    if (m) notify("Matched at iteration: " + currentIteration);
    return m;
  }

  @Override
  public void update() {
    this.currentIteration++;
  }

  @Override
  public CriterionType getType() {
    return CriterionType.ITERATION;
  }

  @Override
  public void reset() {
    this.currentIteration = 0;
  }
}
