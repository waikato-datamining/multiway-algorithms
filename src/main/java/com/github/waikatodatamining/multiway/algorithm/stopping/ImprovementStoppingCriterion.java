package com.github.waikatodatamining.multiway.algorithm.stopping;

import com.github.waikatodatamining.multiway.exceptions.InvalidInputException;
import lombok.extern.slf4j.Slf4j;

/**
 * Stopping criterion that checks the relative improvement is below a given
 * threshold.
 *
 * @author Steven Lang
 */
@Slf4j
public class ImprovementStoppingCriterion extends StoppingCriterion<Double> {

  /** Improvement threshold */
  private double tol;

  /** Old loss */
  private double oldLoss;

  /** Relative improvement in the last iteration */
  private double improvement;

  /**
   * Construct improvement criterion with given threshold.
   *
   * @param tol Improvement tolerance
   */
  public ImprovementStoppingCriterion(double tol) {
    this.tol = tol;
    this.oldLoss = Double.MAX_VALUE;
    this.improvement = Double.MAX_VALUE;
    validateParameters();
  }

  @Override
  public boolean matches() {
    final boolean m = improvement < tol;
    if (m) notify("Last improvement was: " + improvement + "%");
    return m;
  }

  @Override
  public void update(Double newLoss) {
    improvement = Math.abs(oldLoss - newLoss) / oldLoss;
    oldLoss = newLoss;
  }

  @Override
  public CriterionType getType() {
    return CriterionType.IMPROVEMENT;
  }

  @Override
  public void reset() {
    oldLoss = Double.MAX_VALUE;
    improvement = Double.MAX_VALUE;
  }


  @Override
  public void validateParameters() {
    if (tol <= 0) {
      throw new InvalidInputException("Improvement threshold must be greater" +
	" than zero.");
    }
  }
}
