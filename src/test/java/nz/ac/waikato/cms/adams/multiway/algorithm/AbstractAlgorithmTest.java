package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.RegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.IterationCriterion;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public abstract class AbstractAlgorithmTest<T extends AbstractAlgorithm> {

  private static Logger logger = LoggerFactory.getLogger(AbstractAlgorithmTest.class);

  private List<RegressionTestManager> regressionTestManagers = new ArrayList<>();

  protected abstract T constructAlgorithm();

  @Test
  public abstract void testBuildWithNull();

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

  public abstract void setupRegressionTests();

  public void addRegressionTest(RegressionTestManager tm) {
    regressionTestManagers.add(tm);
  }

  @Test
  abstract public void testKill();

}
