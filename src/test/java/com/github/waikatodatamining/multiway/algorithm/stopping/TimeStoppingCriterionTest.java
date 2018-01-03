package com.github.waikatodatamining.multiway.algorithm.stopping;

import com.github.waikatodatamining.multiway.TestUtils;
import com.github.waikatodatamining.multiway.exceptions.InvalidInputException;
import org.apache.commons.lang3.time.StopWatch;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Time stopping criterion testcase.
 *
 * @author Steven Lang
 */
public class TimeStoppingCriterionTest {

  private TimeStoppingCriterion tsc;

  @Before
  public void init() {
    tsc = new TimeStoppingCriterion(1);
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
    try {
      tsc = new TimeStoppingCriterion(10L);
    }
    catch (InvalidInputException iie) {
      fail("Valid parameter has been evaluated as invalid.");
    }

    try {
      tsc = new TimeStoppingCriterion(-10L);
      fail("Invalid parameter has been evaluated as valid.");
    }
    catch (InvalidInputException iie) {
      // Success
    }

    try {
      tsc = new TimeStoppingCriterion(0);
      fail("Invalid parameter has been evaluated as valid.");
    }
    catch (InvalidInputException iie) {
      // Success
    }
  }
}