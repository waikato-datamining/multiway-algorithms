package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.MultiBlockSupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.io.IOException;
import java.util.Map;

import static nz.ac.waikato.cms.adams.multiway.algorithm.regression.SupervisedRegressionTestManager.YHAT;

/**
 * Regression test manager for MultiBlock supervised algorithms.
 *
 * @param <E> Algorithm type
 * @author Steven Lang
 */
public abstract class MultiBlockSupervisedRegressionTestManager<E extends MultiBlockSupervisedAlgorithm> extends RegressionTestManager<E> {

  @Override
  public final String buildAlgorithmWithTestData() throws IOException {
    // Build algorithm input
    Tensor[] tensors = TestUtils.loadSyntheticMultiBlockSupervisedData();
    Tensor[] xblocks = new Tensor[]{tensors[0], tensors[1]};
    Tensor y = tensors[2];
    return algorithm.build(xblocks, y);
  }

  /**
   * Regression test data.
   *
   * @return Regression test data for multiblock supervised algorithms
   */
  @Override
  protected Tensor[] getRegressionTestData() {
    return TestUtils.loadSyntheticMultiBlockSupervisedData();
  }


  @Override
  public Map<String, Tensor> generateResults() {
    Tensor[] tensors = getRegressionTestData();
    Tensor[] xblocks = new Tensor[]{tensors[0], tensors[1]};
    Map<String, Tensor> refs = super.generateResults();
    Tensor Yhat = algorithm.predict(xblocks);
    refs.put(YHAT, Yhat);
    return refs;
  }
}