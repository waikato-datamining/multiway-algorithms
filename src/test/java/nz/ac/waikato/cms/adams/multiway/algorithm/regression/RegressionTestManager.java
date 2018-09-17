package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;
import nz.ac.waikato.cms.adams.multiway.data.DataReader;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.io.File;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

/**
 * Manager that handles regression tests for this algorithm. Specifically it
 * defines where the regression reference results are, how the reference results
 * have to be compared to the new results and how they are saved and loaded.
 *
 * @param <E> Algorithm implementation
 * @author Steven Lang
 */
public abstract class RegressionTestManager<E extends AbstractAlgorithm> {

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
   * Run the regression test, defined by the current algorithm and options that
   * have been set. Previous results that were stored in the regression
   * reference files will be compared to the new result.
   *
   * @return True if test is successful (new results equal reference results)
   * @throws IOException Could not access the reference data
   */
  public abstract boolean run() throws IOException;

  /**
   * Check if reference exists boolean.
   *
   * @return the boolean
   */
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

  /**
   * Save new reference.
   *
   * @throws IOException the io exception
   */
  public abstract void saveNewReference() throws IOException;

  /**
   * Load the reference objects from the given directory at {@link
   * RegressionTestManager#getRegressionReferenceDirectory()}*.
   *
   * @return Reference objects
   * @throws IOException Could not access directory
   */
  public abstract Map<String, Tensor> loadReference() throws IOException;

  /**
   * Get path to the regression reference directory.
   *
   * @return Regression reference directory path
   */
  private String getRegressionReferenceDirectory() {
    String path = algorithm.getClass().getSimpleName()
      .replaceAll("\\.", File.separator) // Replace "." with "/"
      .replaceAll("Test", ""); // Remove "Test"
    String base = "src/test/resources/data/regression/ref";
    String refDir = Paths.get(base, path, options).toString();
    return refDir;
  }

  /**
   * Gets reference file path.
   *
   * @param specifier the specifier
   * @return the reference file path
   */
  protected String getReferenceFilePath(String specifier) {
    return getRegressionReferenceDirectory() + "/ref" + "-" + specifier + ".csv";
  }

  /**
   * Gets algorithm.
   *
   * @return the algorithm
   */
  public E getAlgorithm() {
    return algorithm;
  }

  /**
   * Sets algorithm.
   *
   * @param algorithm the algorithm
   */
  public void setAlgorithm(E algorithm) {
    this.algorithm = algorithm;
  }

  /**
   * Gets options.
   *
   * @return the options
   */
  public String getOptions() {
    return options;
  }

  /**
   * Sets options.
   *
   * @param options the options
   */
  public void setOptions(String options) {
    this.options = options;
  }

  /**
   * Load the test data used for the regression tests.
   *
   * @return Array of Tensors containing the regression test data
   */
  protected abstract Tensor[] getRegressionTestData();

  /**
   * Load a reference matrix.
   *
   * @param refSpecifier Reference specifier
   * @return Tensor containing the matrix
   * @throws IOException Could not read from path
   */
  protected Tensor loadMatrix(String refSpecifier) throws IOException {
    return Tensor.create(DataReader.readMatrixCsv(getReferenceFilePath(refSpecifier), ","));
  }

  /**
   * Save a reference matrix.
   *
   * @param refSpecifier Reference specifier
   * @param matrix       Tensor containing the matrix which is to be saved
   * @throws IOException Could not read from path
   */
  protected void saveMatrix(String refSpecifier, Tensor matrix) throws IOException {
    DataReader.writeMatrixCsv(matrix.toArray2d(), getReferenceFilePath(refSpecifier), ",");
  }
}
