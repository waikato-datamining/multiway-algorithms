package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

import nz.ac.waikato.cms.adams.multiway.exceptions.InvalidInputException;
import lombok.extern.slf4j.Slf4j;

/**
 * Stopping criterion that checks if a maximum number of iterations is reached.
 *
 * @author Steven Lang
 */
@Slf4j
public class IterationStoppingCriterion extends StoppingCriterion<Integer> {

  /** Current iteration */
  private int currentIteration;

  /** Maximum number of iterations */
  private final int maxIterations;

  /**
   * Construct an iteration stopping criterion with a maximum number of
   * iterations.
   *
   * @param maxIterations Maximum number of iterations
   */
  public IterationStoppingCriterion(int maxIterations) {
    this.currentIteration = 0;
    this.maxIterations = maxIterations;
    validateParameters();
  }

  @Override
  public boolean matches() {
    final boolean m = currentIteration >= maxIterations;
    if (m) notify("Stopped at iteration: " + currentIteration);
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

  @Override
  public void validateParameters() {
    if (maxIterations <= 0){
      throw new InvalidInputException("Maximum number of iterations must be greater" +
        " than zero.");
    }
  }
}
