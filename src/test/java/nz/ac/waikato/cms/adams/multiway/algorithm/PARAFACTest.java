package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.PARAFAC.Initialization;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.PARAFACRegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.DataReader;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.exceptions.ModelNotBuiltException;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * {@link PARAFAC} algorithm testcase.
 *
 * @author Steven Lang
 */
public class PARAFACTest extends AbstractUnsupervisedAlgorithmTest<PARAFAC> {

  private static final int numComponents = 2;

  private static final int I = 5;

  private static final int J = 4;

  private static final int K = 3;

  private static final int numStarts = 3;

  private static final int maxIter = 10;

  private PARAFAC pf;

  @Before
  public void initializeTest() {
    pf = new PARAFAC();
    pf.setNumComponents(numComponents);
    pf.setNumStarts(numStarts);
    pf.setInitMethod(Initialization.RANDOM);
    pf.addStoppingCriterion(CriterionUtils.iterations(maxIter));
  }

  @Test
  public void testGetLoadings() {

    pf.build(TestUtils.generateRandomTensor(I, J, K));
    final Map<String, Tensor> loadingMatrices = pf.getLoadingMatrices();
    assertEquals(3, loadingMatrices.size());
    assertEquals(I, loadingMatrices.get("A").size(0));
    assertEquals(J, loadingMatrices.get("B").size(0));
    assertEquals(K, loadingMatrices.get("C").size(0));
    assertEquals(2, loadingMatrices.get("A").order());
    assertEquals(2, loadingMatrices.get("B").order());
    assertEquals(2, loadingMatrices.get("C").order());
    assertEquals(numComponents, loadingMatrices.get("A").size(1));
    assertEquals(numComponents, loadingMatrices.get("B").size(1));
    assertEquals(numComponents, loadingMatrices.get("C").size(1));
  }

  @Test
  public void testGetLossHistories() {
    pf.build(TestUtils.generateRandomTensor(I, J, K));
    final List<List<Double>> lossHistory = pf.getLossHistory();

    assertEquals(numStarts, lossHistory.size());
    lossHistory.forEach(h -> assertEquals(maxIter, h.size()));
  }

  @Test(expected = ModelNotBuiltException.class)
  public void testFilterUnbuiltModel() {
    pf.filter(Tensor.create(1));
  }


  @Test
  public void testFilterOutputDims() {
    final Tensor data = TestUtils.generateRandomTensor(10, 4, 5);
    pf.build(data);
    final Tensor testData = TestUtils.generateRandomTensor(10, 4, 5);
    final Tensor transformed = pf.filter(testData);

    assertEquals(testData.size(0), transformed.size(0));
    assertEquals(numComponents, transformed.size(1));
    assertEquals(2, transformed.order());
  }

  @Override
  protected PARAFAC constructAlgorithm() {
    return new PARAFAC();
  }


  @Override
  public void setupRegressionTests() {
    // Default
    PARAFAC p0 = new PARAFAC();
    p0.addStoppingCriterion(CriterionUtils.iterations(10));
    addRegressionTest(p0, "default");

    // SVD
    PARAFAC p1 = new PARAFAC();
    p1.setInitMethod(Initialization.SVD);
    p1.addStoppingCriterion(CriterionUtils.iterations(10));
    addRegressionTest(p1, Initialization.SVD.toString());

    // RANDOM_ORTH
    PARAFAC p2= new PARAFAC();
    p2.setInitMethod(Initialization.RANDOM_ORTHOGONALIZED);
    p2.addStoppingCriterion(CriterionUtils.iterations(10));
    addRegressionTest(p2, Initialization.RANDOM_ORTHOGONALIZED.toString());
  }

  public void addRegressionTest(PARAFAC pf, String options) {
    PARAFACRegressionTestManager regtest = new PARAFACRegressionTestManager();
    regtest.setAlgorithm(pf);
    regtest.setOptions(options);
    addRegressionTest(regtest);
  }



}
