package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.DataReader;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link MultiLinearPLS} algorithm testcase.
 *
 * @author Steven Lang
 */
public class MultiLinearPLSTest extends
  AbstractSupervisedAlgorithmTest<MultiLinearPLS> {

  @Test
  public void doBuild() {
    final Tensor X = TestUtils.generateRandomTensor(10, 5, 3);
    final Tensor Y = TestUtils.generateRandomMatrix(10, 2);
    MultiLinearPLS mpls = new MultiLinearPLS();
    mpls.addStoppingCriterion(CriterionUtils.iterations(10));
    final String err = mpls.build(X, Y);
    assertTrue("Error was: " + err, err == null);
  }

  @Test
  public void filter() {
    final int numTestRows = 55;
    final int F = 9;
    final Tensor X = TestUtils.generateRandomTensor(10, 5, 3);
    final Tensor X2 = TestUtils.generateRandomTensor(numTestRows, 5, 3);
    final Tensor Y = TestUtils.generateRandomMatrix(10, 2);
    MultiLinearPLS mpls = new MultiLinearPLS();
    mpls.setNumComponents(F);
    mpls.addStoppingCriterion(CriterionUtils.iterations(10));
    mpls.build(X, Y);
    final Tensor filter = mpls.filter(X2);


    assertEquals(2, filter.order());
    assertEquals(numTestRows, filter.size(0));
    assertEquals(F, filter.size(1));
  }

  @Test
  public void predict() throws IOException {
    String xpath = "src/test/resources/data/X-threeway.csv";
    String ypath = "src/test/resources/data/Y-multitarget.csv";

    final double[][][] Xdata = DataReader.read3WaySparse(xpath, " ", 3, false);
    final double[][] Ydata = DataReader.readSparseMatrix(ypath, " ", false);
    Tensor X = Tensor.create(Xdata);
    Tensor Y = Tensor.create(Ydata);

    MultiLinearPLS mpls = new MultiLinearPLS();
    mpls.setNumComponents(35);
    mpls.build(X, Y);
    // Dataset is known from previous tests to achieve < 1 MSE
    // If that changes, something must be odd with the PLS code
    final Tensor Ypred = mpls.predict(X);
    final double mse = MathUtils.meanSquaredError(Y, Ypred);
    assertTrue("MSE is known to be below 1.0 for this dataset with " +
      "a working implementation", mse < 1d);
  }

  @Override
  protected MultiLinearPLS constructAlgorithm() {
    return new MultiLinearPLS();
  }

  @Override
  public void setupRegressionTests() {

  }
}