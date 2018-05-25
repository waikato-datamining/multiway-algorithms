package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

/**
 * Criterion that immediately ends execution.
 *
 * @author Steven Lang
 */
public class KillCriterion extends Criterion {

  private static final long serialVersionUID = -7761506389193605767L;

  @Override
  public boolean matches() {
    notify("Stopping execution now.");
    return true;
  }

  @Override
  public CriterionType getType() {
    return CriterionType.KILL;
  }

  @Override
  public void reset() {
  }

  @Override
  public void update() {}
}
