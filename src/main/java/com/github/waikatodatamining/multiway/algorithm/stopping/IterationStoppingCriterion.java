package com.github.waikatodatamining.multiway.algorithm.stopping;

import com.github.waikatodatamining.multiway.exceptions.InvalidInputException;

/**
 * Stopping criterion that checks if a maximum number of iterations is reached.
 *
 * @author Steven Lang
 */
public class IterationStoppingCriterion implements StoppingCriterion<Integer> {

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
    return currentIteration >= maxIterations;
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
