package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.MultiLinearPLSRegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.RegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.SONPLSRegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;

/**
 * Tests the {@link SONPLS} algorithm.
 *
 * @author Steven Lang
 */
public class SONPLSTest extends AbstractMultiBlockSupervisedAlgorithmTest<SONPLS> {

  @Override
  protected SONPLS constructAlgorithm() {
    return new SONPLS();
  }

  @Override
  public void setupRegressionTests() {
    SONPLS pls = constructAlgorithm();
    pls.addStoppingCriterion(CriterionUtils.iterations(10));
    addRegressionTest(pls, "default");

    pls = constructAlgorithm();
    pls.setAutoNumComponents(false);
    pls.setNumComponents(new int[]{2, 2});
    pls.addStoppingCriterion(CriterionUtils.iterations(10));
    addRegressionTest(pls, "autoNumComponentsFalse");

    pls = constructAlgorithm();
    pls.setStandardizeY(false);
    pls.addStoppingCriterion(CriterionUtils.iterations(10));
    addRegressionTest(pls, "standardizeYfalse");

  }

  @Override
  public RegressionTestManager<? extends AbstractAlgorithm, ?> createRegressionTestManager() {
    return new SONPLSRegressionTestManager();
  }
}
