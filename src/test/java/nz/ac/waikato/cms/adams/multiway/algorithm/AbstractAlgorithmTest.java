package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.RegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.IterationCriterion;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Abstract algorithm test.
 *
 * @param <T> Abstract algorithm implementation
 * @author Steven Lang
 */
public abstract class AbstractAlgorithmTest<T extends AbstractAlgorithm> {

  /** Logger */
  private static Logger logger = LoggerFactory.getLogger(AbstractAlgorithmTest.class);

  /**
   * Local regression test manager list for algorithm implementation of type
   * {@link T}.
   */
  private List<RegressionTestManager<? extends AbstractAlgorithm>> regressionTestManagers = new ArrayList<>();

  /**
   * Construct an instance of {@link T}.
   *
   * @return Instance of {@link T}
   */
  protected abstract T constructAlgorithm();

  /**
   * Test calling build() method with null as arguments.
   */
  @Test
  public abstract void testBuildWithNull();

  /**
   * Run all regression tests that have been setup in {@link
   * AbstractAlgorithmTest#setupRegressionTests()}.
   *
   * @throws IOException Something went wrong reading or writing reference
   *                     files.
   */
  @Test
  public final void runRegressionTests() throws IOException {
    setupRegressionTests();
    List<String> failed = new ArrayList<>();
    for (RegressionTestManager testManager : regressionTestManagers) {
      String options = testManager.getOptions();
      String simpleName = testManager.getAlgorithm().getClass().getSimpleName();
      logger.info("Running regression test for: {} ({})",
	simpleName,
	options);
      boolean success = testManager.run();
      if (!success) {
	failed.add(simpleName + "::" + options);
      }
    }

    if (!failed.isEmpty()) {
      logger.error("The following regression tests have failed:" + System.lineSeparator());
      failed.forEach(logger::error);
      fail();
    }
  }

  /**
   * Construct a thread that kills the algorithm after one second and asserts,
   * that the algorithm has preemptive stopped.
   *
   * @param alg      Algorithm to kill
   * @param maxIters Maximum number of iterations
   * @return Thread that kill the given algorithm and asserts its success
   */
  protected Thread getAlgorithmKillingThread(T alg, int maxIters) {
    return new Thread(() -> {
      try {
	Thread.sleep(1000);
	alg.stopExecution();
	IterationCriterion itercrit = (IterationCriterion) Arrays
	  .stream(alg.getStoppingCriteria())
	  .filter(c -> c.getType().equals(CriterionType.ITERATION))
	  .findAny()
	  .get();
	int currentIteration = itercrit.getCurrentIteration();
	assertTrue(currentIteration < maxIters);
      }
      catch (InterruptedException e) {
	e.printStackTrace();
      }
    });
  }

  /**
   * Setup regression tests by repeatedly calling {@link
   * AbstractAlgorithmTest#addRegressionTest(AbstractAlgorithm, String)} with a
   * specific algorithm setup and a string specifier describing the setup.
   */
  public abstract void setupRegressionTests();

  /**
   * Add a regression test.
   *
   * @param alg     Algorithm setup
   * @param options String specifier
   */
  public void addRegressionTest(T alg, String options) {
    RegressionTestManager<AbstractAlgorithm> regTest = (RegressionTestManager<AbstractAlgorithm>) createRegressionTestManager();
    regTest.setAlgorithm(alg);
    regTest.setOptions(options);
    regressionTestManagers.add(regTest);
  }

  /**
   * Get an instance of the current regression test manager which is to be
   * used.
   *
   * @return Instance of {@link RegressionTestManager} implementation
   */
  public abstract RegressionTestManager<? extends AbstractAlgorithm> createRegressionTestManager();

  /**
   * Tests the {@link AbstractAlgorithm#stopExecution()} method.
   */
  @Test
  abstract public void testStopExecution();

}
