package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

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
