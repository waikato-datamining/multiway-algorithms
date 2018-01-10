package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Time stopping criterion testcase.
 *
 * @author Steven Lang
 */
public class TimeCriterionTest {

  private TimeCriterion tsc;

  @Before
  public void init() {
    tsc = CriterionUtils.time(1);
  }

  @Test
  public void matches() throws InterruptedException {
    assertFalse(tsc.matches());
    Thread.sleep(1500);
    tsc.update();
    assertTrue(tsc.matches());
  }

  @Test
  public void update() throws NoSuchFieldException, IllegalAccessException, InterruptedException {
    tsc.reset();
    tsc.matches();
    tsc.update();
    final long t0 = TestUtils.getField(tsc, "secondsElapsed");
    Thread.sleep(1000);
    tsc.update();
    final long t1 = TestUtils.getField(tsc, "secondsElapsed");
    assertTrue(t1 > t0);
    assertEquals(1, t1 - t0);
  }

  @Test
  public void getType() {
    assertEquals(CriterionType.TIME, tsc.getType());
  }

  @Test
  public void reset() throws NoSuchFieldException, IllegalAccessException {
    tsc.reset();
    final StopWatch sw = TestUtils.getField(tsc, "sw");
    assertTrue(sw.isStopped());
  }

  @Test
  public void validateParameters() {
    tsc = CriterionUtils.time(10L);
    tsc.setMaxSeconds(-1);
    assertEquals(10L, tsc.getMaxSeconds());
  }
}