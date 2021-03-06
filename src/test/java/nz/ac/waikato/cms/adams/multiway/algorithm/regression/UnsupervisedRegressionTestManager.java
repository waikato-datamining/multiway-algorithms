package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.UnsupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.io.IOException;

/**
 * Regression test manager for unsupervised algorithms.
 *
 * @param <E> Unsupervised algorithm implementation
 * @author Steven Lang
 */
public abstract class UnsupervisedRegressionTestManager<E extends UnsupervisedAlgorithm> extends RegressionTestManager<E> {

  @Override
  public final String buildAlgorithmWithTestData() throws IOException {
    Tensor data = getRegressionTestData()[0];
    return algorithm.build(data);
  }

  public Tensor[] getRegressionTestData() {
    return new Tensor[]{TestUtils.loadFluorescenceData()};
  }
}