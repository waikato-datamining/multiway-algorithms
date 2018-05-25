package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

/**
 * Utility class for convenient creation of criteria.
 *
 * @author Steven Lang
 */
public class CriterionUtils {

  /**
   * Create a new iteration criterion.
   *
   * @param maxIters Number of maximum iterations.
   * @return Iteration criterion
   */
  public static IterationCriterion iterations(int maxIters) {
    IterationCriterion c = new IterationCriterion();
    c.setMaxIterations(maxIters);
    return c;
  }


  /**
   * Create a new time criterion.
   *
   * @param maxSeconds Number of maximum iterations.
   * @return Time criterion
   */
  public static TimeCriterion time(long maxSeconds) {
    TimeCriterion c = new TimeCriterion();
    c.setMaxSeconds(maxSeconds);
    return c;
  }

  /**
   * Create a new improvement criterion.
   *
   * @param tol Improvement tolerance.
   * @return Improvement criterion
   */
  public static ImprovementCriterion improvement(double tol) {
    ImprovementCriterion c = new ImprovementCriterion();
    c.setTol(tol);
    return c;
  }

  /**
   * Create a new kill criterion
   *
   * @return Kill criterion
   */
  public static KillCriterion kill() {
    return new KillCriterion();
  }
}
