package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.RegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.SONPLSRegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Tests the {@link SONPLS} algorithm.
 *
 * @author Steven Lang
 */
public class SONPLSTest extends AbstractMultiBlockSupervisedAlgorithmTest<SONPLS> {

  @Test
  public void testBuildMixedRankBlocks(){
    // Generate blocks of different ranks
    int[] shape1 = {10, 5, 4};
    int[] shape2 = {10, 4, 1};
    int[] shape3 = {10, 2};
    int[] shape4 = {10, 1, 5};

    Tensor x1 = Tensor.create(Nd4j.randn(shape1, 1));
    Tensor x2 = Tensor.create(Nd4j.randn(shape2, 2));
    Tensor x3 = Tensor.create(Nd4j.randn(shape3, 3));
    Tensor x4 = Tensor.create(Nd4j.randn(shape4, 4));


    Tensor[] x = {x1, x2, x3, x4};
    Tensor y = Tensor.create(Nd4j.randn(10, 2, 0));

    SONPLS sonpls = constructAlgorithm();
    String build = sonpls.build(x, y);

    Assert.assertNull("Error: " + build, build);
  }

  @Override
  protected SONPLS constructAlgorithm() {
    SONPLS sonpls = new SONPLS();
    sonpls.addStoppingCriterion(CriterionUtils.iterations(10));
    return sonpls;
  }

  @Override
  public void setupRegressionTests() {
    SONPLS pls = constructAlgorithm();
    addRegressionTest(pls, "default");

    pls = constructAlgorithm();
    pls.setAutoNumComponents(false);
    pls.setNumComponents(new int[]{2, 2});
    addRegressionTest(pls, "autoNumComponentsFalse");

    pls = constructAlgorithm();
    pls.setStandardizeY(false);
    addRegressionTest(pls, "standardizeYfalse");

  }

  @Override
  public RegressionTestManager<? extends AbstractAlgorithm> createRegressionTestManager() {
    return new SONPLSRegressionTestManager();
  }
}
