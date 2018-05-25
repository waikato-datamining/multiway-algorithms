package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;
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
      logger.info("Running regression test for: {} ({})",
	testManager.algorithm.getClass().getSimpleName(),
	testManager.options);
      boolean success = testManager.run();
      if (!success) {
	failed.add(testManager.algorithm.getClass().getSimpleName() + "::" + testManager.options);
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

  /**
   * Manager that handles regression tests for this algorithm. Specifically it
   * defines where the regression reference results are, how the reference
   * results have to be compared to the new results and how they are saved and
   * loaded.
   *
   * @param <E> Algorithm implementation
   * @param <R> Reference object type
   */
  public abstract class RegressionTestManager<E extends AbstractAlgorithm, R> {

    /**
     * Algorithm object which is to be tested.
     */
    protected E algorithm;

    /**
     * Options string that describes with which options the algorithm has been
     * executed.
     */
    protected String options;

    /**
     * Run the regression test, defined by the current algorithm and options
     * that have been set. Previous results that were stored in the regression
     * reference files will be compared to the new result.
     *
     * @return True if test is successful (new results equal reference results)
     * @throws IOException Could not access the reference data
     */
    public abstract boolean run() throws IOException;

    public boolean checkIfReferenceExists() {
      try (DirectoryStream<Path> dirStream = Files.newDirectoryStream(Paths.get(getRegressionReferenceDirectory()))) {
	return dirStream.iterator().hasNext();
      }
      catch (IOException e) {
	return false;
      }
    }

    /**
     * Check if the generated results are equal to the prior reference result.
     *
     * @return True if equal
     * @throws IOException Could not access the reference data
     */
    public abstract boolean resultEqualsReference() throws IOException;

    public abstract void saveNewReference() throws IOException;

    public abstract R loadReference() throws IOException;

    public abstract String getRegressionReferenceDirectory();

    public String getRegressionReferenceBaseDirectory() {
      return "src/test/resources/data/regression";
    }

    protected String getReferenceFilePath(String specifier) {
      return getRegressionReferenceDirectory() + "/ref" + "-" + specifier + ".csv";
    }

    public E getAlgorithm() {
      return algorithm;
    }

    public void setAlgorithm(E algorithm) {
      this.algorithm = algorithm;
    }

    public String getOptions() {
      return options;
    }

    public void setOptions(String options) {
      this.options = options;
    }
  }
}
