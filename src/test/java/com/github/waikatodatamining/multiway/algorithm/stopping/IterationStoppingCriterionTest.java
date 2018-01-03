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
 * Iteration stopping criterion testcase.
 *
 * @author Steven Lang
 */
public class IterationStoppingCriterionTest {

  private IterationStoppingCriterion isc;

  @Before
  public void init() {
    isc = new IterationStoppingCriterion(10);
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
  public void update() throws NoSuchFieldException, IllegalAccessException {
    int currIt = TestUtils.<Integer>getField(isc, "currentIteration");
    assertEquals(0, currIt);
    isc.update();
    currIt = TestUtils.<Integer>getField(isc, "currentIteration");
    assertEquals(1, currIt);
  }

  @Test
  public void getType() {
    assertEquals(CriterionType.ITERATION, isc.getType());
  }

  @Test
  public void reset() throws NoSuchFieldException, IllegalAccessException {
    int currIt = TestUtils.<Integer>getField(isc, "currentIteration");
    assertEquals(0, currIt);
  }

  @Test
  public void validateParameters() {
    try {
      isc = new IterationStoppingCriterion(10);
    }
    catch (InvalidInputException iie) {
      fail("Valid parameter has been evaluated as invalid.");
    }

    try {
      isc = new IterationStoppingCriterion(-10);
      fail("Invalid parameter has been evaluated as valid.");
    }
    catch (InvalidInputException iie) {
      // Success
    }

    try {
      isc = new IterationStoppingCriterion(0);
      fail("Invalid parameter has been evaluated as valid.");
    }
    catch (InvalidInputException iie) {
      // Success
    }
  }
}