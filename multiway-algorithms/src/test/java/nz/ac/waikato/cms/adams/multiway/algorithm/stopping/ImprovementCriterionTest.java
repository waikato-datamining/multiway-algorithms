package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Improvement stopping criterion testcase.
 *
 * @author Steven Lang
 */
public class ImprovementCriterionTest {

  private ImprovementCriterion isc;

  @Before
  public void init() {
    isc = CriterionUtils.improvement(10E-5);
  }

  @Test
  public void matches() {
    isc.update(10d);
    isc.update(5d);
    assertFalse(isc.matches());

    isc.update(5d);
    assertTrue(isc.matches());
  }

  @Test
  public void update() {
    final double newLoss = 10d;
    isc.update(newLoss);
    final double oldLoss = isc.oldLoss;
    assertEquals(newLoss, oldLoss, 10e-5);

    isc.update(5d);
    final double improvementExp = 0.5d;
    final double improvementActual = isc.improvement;
    assertEquals(improvementExp, improvementActual, 10e-5);
  }

  @Test
  public void getType() {
    assertEquals(CriterionType.IMPROVEMENT, isc.getType());
  }

  @Test
  public void reset() {
    isc.reset();
    assertEquals(Double.MAX_VALUE, isc.oldLoss, 10e-5);
    assertEquals(Double.MAX_VALUE, isc.improvement, 10e-5);
  }

  @Test
  public void validateParameters() {
    isc = CriterionUtils.improvement(0.5);
    isc.setTol(-0.5);
    assertEquals(0.5, isc.getTol(), 10E-5);
  }
}