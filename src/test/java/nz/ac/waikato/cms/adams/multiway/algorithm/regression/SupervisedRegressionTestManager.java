package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.SupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.io.IOException;
import java.util.Map;

/**
 * Regression test manager for supervised algorithms.
 *
 * @param <E> Supervised algorithm implementation
 * @author Steven Lang
 */
public abstract class SupervisedRegressionTestManager<E extends SupervisedAlgorithm> extends RegressionTestManager<E> {

  /** Yhat specifier */
  protected static final String YHAT = "Yhat";

  @Override
  public final String buildAlgorithmWithTestData() {
    Tensor[] data = getRegressionTestData();
    return algorithm.build(data[0], data[1]);
  }

  @Override
  public Tensor[] getRegressionTestData() {
    return TestUtils.loadSyntheticSupervisedData();
  }

  @Override
  public Map<String, Tensor> generateResults() {
    Map<String, Tensor> refs = super.generateResults();
    Tensor Yhat = algorithm.predict(getRegressionTestData()[0]);
    refs.put(YHAT, Yhat);
    return refs;
  }
}