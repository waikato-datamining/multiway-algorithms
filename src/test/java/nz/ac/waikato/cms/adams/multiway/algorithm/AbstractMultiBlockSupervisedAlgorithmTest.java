package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.MultiBlockSupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

/**
 * Abstract class for MultiBlock supervised algorithms.
 *
 * @param <T> MultiBlockSupervisedAlgorithm implementation
 * @author Steven Lang
 */
public abstract class AbstractMultiBlockSupervisedAlgorithmTest<T extends
  MultiBlockSupervisedAlgorithm> extends AbstractAlgorithmTest<T> {


  @Test
  public void testBuildWithNull() {
    assertNotNull(constructAlgorithm().build(null, null));
  }

  @Test
  public void testKill() {
    Tensor[] tensors = TestUtils.loadSyntheticMultiBlockSupervisedData();
    Tensor[] xblocks = new Tensor[]{tensors[0], tensors[1]};
    Tensor y = tensors[2];
    T alg = constructAlgorithm();
    int maxIters = 10000000;
    alg.addStoppingCriterion(CriterionUtils.iterations(maxIters));
    Thread t = getAlgorithmKillingThread(alg, maxIters);
    t.start();
    alg.build(xblocks, y);
  }


}
