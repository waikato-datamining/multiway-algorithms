package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.UnsupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

/**
 * Abstract unsupervised algorithm test.
 *
 * @param <T> Abstract unsupervised algorithm implementation
 * @author Steven Lang
 */
public abstract class AbstractUnsupervisedAlgorithmTest<T extends
  UnsupervisedAlgorithm> extends AbstractAlgorithmTest<T> {

  @Override
  public void testBuildWithNull() {
    assertNotNull(constructAlgorithm().build(null));
  }

  @Test
  public void testStopExecution() {
    Tensor data = TestUtils.generateRandomTensor(10, 10, 2);
    T alg = constructAlgorithm();
    int maxIters = 100000;
    alg.addStoppingCriterion(CriterionUtils.iterations(maxIters));
    Thread t = getAlgorithmKillingThread(alg, maxIters);
    t.start();
    alg.build(data);
  }
}
