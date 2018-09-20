package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.MNPLSRegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.RegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.twoway.MNPLS;

/**
 * {@link MNPLS} algorithm testcase.
 *
 * @author Steven Lang
 */
public class MNPLSTest extends
  AbstractSupervisedAlgorithmTest<MNPLS> {

  @Override
  protected MNPLS constructAlgorithm() {
    return new MNPLS();
  }

  @Override
  public void setupRegressionTests() {
    MNPLS pls = constructAlgorithm();
    addRegressionTest(pls, "default");

    pls = constructAlgorithm();
    pls.setStandardizeY(false);
    addRegressionTest(pls, "standardizeYfalse");
  }

  @Override
  public RegressionTestManager<? extends AbstractAlgorithm> createRegressionTestManager() {
    return new MNPLSRegressionTestManager();
  }
}