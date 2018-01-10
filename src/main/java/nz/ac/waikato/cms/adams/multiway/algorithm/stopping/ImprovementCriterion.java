package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Criterion that checks the relative improvement is below a given
 * threshold.
 *
 * @author Steven Lang
 */
public class ImprovementCriterion extends Criterion<Double> {

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(ImprovementCriterion.class);

  /** Serial version UID */
  private static final long serialVersionUID = -7578999397822790016L;

  /** Improvement tolerance */
  protected double tol;

  /** Old loss */
  protected double oldLoss;

  /** Relative improvement in the last iteration */
  protected double improvement;


  @Override
  protected void initialize() {
    super.initialize();
    this.tol = 1E-8;
    this.oldLoss = Double.MAX_VALUE;
    this.improvement = Double.MAX_VALUE;

  }

  /**
   * Get the improvement tolerance.
   *
   * @return Improvement tolerance
   */
  public double getTol() {
    return tol;
  }


  /**
   * Set the improvement tolerance.
   *
   * @param tol Improvement tolerance
   */
  public void setTol(double tol) {
    if (tol <= 0) {
      log.warn("Improvement threshold must be greater" +
	" than zero.");
    }
    else {
      this.tol = tol;
    }
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
}
