package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.SupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.regression.RegressionTestManager;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

public abstract class AbstractSupervisedAlgorithmTest<T extends
  SupervisedAlgorithm> extends AbstractAlgorithmTest<T> {


  @Test
  public void testBuildWithNull() {
    assertNotNull(constructAlgorithm().build(null, null));
  }

  @Test
  public void testKill() {
    Tensor data = TestUtils.generateRandomTensor(10, 10, 2);
    Tensor y = TestUtils.generateRandomMatrix(10, 2);
    T alg = constructAlgorithm();
    int maxIters = 10000000;
    alg.addStoppingCriterion(CriterionUtils.iterations(maxIters));
    Thread t = getAlgorithmKillingThread(alg, maxIters);
    t.start();
    alg.build(data, y);
  }


}
