package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.UnsupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.Criterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.IterationCriterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.KillCriterion;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public abstract class AbstractUnsupervisedAlgorithmTest<T extends
  UnsupervisedAlgorithm> extends AbstractAlgorithmTest<T> {

  @Override
  public void testBuildWithNull() {
    assertNotNull(constructAlgorithm().build(null));
  }

  public Tensor getRegressionTestData() {
    return TestUtils.loadRegressionTestData();
  }

  @Test
  public void testKill() {
    Tensor data = TestUtils.generateRandomTensor(10, 10, 2);
    T alg = constructAlgorithm();
    int maxIters = 100000;
    alg.addStoppingCriterion(CriterionUtils.iterations(maxIters));
    Thread t = getAlgorithmKillingThread(alg, maxIters);
    t.start();
    alg.build(data);
  }


  public abstract class UnsupervisedRegressionTestManager<E extends UnsupervisedAlgorithm, R> extends RegressionTestManager<E, R> {

    @Override
    public final boolean run() throws IOException {
      Tensor data = getRegressionTestData();
      algorithm.build(data);
      if (checkIfReferenceExists()) {
	return resultEqualsReference();
      }
      else {
	saveNewReference();
	return true;
      }
    }

    @Override
    public String getRegressionReferenceDirectory() {
      return getRegressionReferenceBaseDirectory() + "/unsupervised/fluorescence/ref";
    }
  }
}
