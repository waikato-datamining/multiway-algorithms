package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.PLS2RegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.RegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.twoway.PLS2;

/**
 * {@link PLS2} algorithm testcase.
 *
 * @author Steven Lang
 */
public class PLS2Test extends
  AbstractSupervisedAlgorithmTest<PLS2> {

  @Override
  protected PLS2 constructAlgorithm() {
    return new PLS2();
  }

  @Override
  public void setupRegressionTests() {
    PLS2 pls = constructAlgorithm();
    addRegressionTest(pls, "default");

    pls = constructAlgorithm();
    pls.setStandardizeY(false);
    addRegressionTest(pls, "standardizeYfalse");
  }

  @Override
  public RegressionTestManager<? extends AbstractAlgorithm> createRegressionTestManager() {
    return new PLS2RegressionTestManager();
  }
}