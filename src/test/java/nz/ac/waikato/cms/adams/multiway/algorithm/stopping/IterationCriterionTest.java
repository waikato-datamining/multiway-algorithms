package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Iteration stopping criterion testcase.
 *
 * @author Steven Lang
 */
public class IterationCriterionTest {

  private IterationCriterion isc;

  @Before
  public void init() {
    isc = CriterionUtils.iterations(10);
  }

  @Test
  public void matches() {
    assertFalse(isc.matches());
    for (int i = 0; i < 10; i++) {
      isc.update();
    }
    assertTrue(isc.matches());
  }

  @Test
  public void update() {
    int currIt = isc.currentIteration;
    assertEquals(0, currIt);
    isc.update();
    currIt = isc.currentIteration;
    assertEquals(1, currIt);
  }

  @Test
  public void getType() {
    assertEquals(CriterionType.ITERATION, isc.getType());
  }

  @Test
  public void reset() {
    int currIt = isc.currentIteration;
    assertEquals(0, currIt);
  }

  @Test
  public void validateParameters() {
    isc = CriterionUtils.iterations(10);
    isc.setMaxIterations(-1);
    assertEquals(10, isc.getMaxIterations());
  }
}