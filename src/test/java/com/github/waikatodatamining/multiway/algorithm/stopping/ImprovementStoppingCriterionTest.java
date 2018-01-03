package com.github.waikatodatamining.multiway.algorithm.stopping;

import com.github.waikatodatamining.multiway.TestUtils;
import com.github.waikatodatamining.multiway.exceptions.InvalidInputException;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Improvement stopping criterion testcase.
 *
 * @author Steven Lang
 */
public class ImprovementStoppingCriterionTest {

  private ImprovementStoppingCriterion isc;

  @Before
  public void init() {
    isc = new ImprovementStoppingCriterion(10e-5);
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
  public void update() throws NoSuchFieldException, IllegalAccessException {
    final double newLoss = 10d;
    isc.update(newLoss);
    final double oldLoss = TestUtils.<Double>getField(isc, "oldLoss");
    assertEquals(newLoss, oldLoss, 10e-5);

    isc.update(5d);
    final double improvementExp = 0.5d;
    final double improvementActual = TestUtils.<Double>getField(isc, "improvement");
    assertEquals(improvementExp, improvementActual, 10e-5);
  }

  @Test
  public void getType() {
    assertEquals(CriterionType.IMPROVEMENT, isc.getType());
  }

  @Test
  public void reset() throws NoSuchFieldException, IllegalAccessException {
    isc.reset();
    assertEquals(Double.MAX_VALUE, TestUtils.<Double>getField(isc, "oldLoss"), 10e-5);
    assertEquals(Double.MAX_VALUE, TestUtils.<Double>getField(isc, "improvement"), 10e-5);
  }

  @Test
  public void validateParameters() {
    try {
      isc = new ImprovementStoppingCriterion(10d);
    }
    catch (InvalidInputException iie) {
      fail("Valid parameter has been evaluated as invalid.");
    }

    try {
      isc = new ImprovementStoppingCriterion(-10d);
      fail("Invalid parameter has been evaluated as valid.");
    }
    catch (InvalidInputException iie) {
      // Success
    }

    try {
      isc = new ImprovementStoppingCriterion(0);
      fail("Invalid parameter has been evaluated as valid.");
    }
    catch (InvalidInputException iie) {
      // Success
    }
  }
}