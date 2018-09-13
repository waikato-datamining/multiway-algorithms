package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.SupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.io.IOException;

public abstract class SupervisedRegressionTestManager<E extends SupervisedAlgorithm, R> extends RegressionTestManager<E, R> {

  @Override
  public final boolean run() throws IOException {
    Tensor[] data = getRegressionTestData();
    algorithm.build(data[0], data[1]);
    if (checkIfReferenceExists()) {
      return resultEqualsReference();
    }
    else {
      saveNewReference();
      return true;
    }
  }

  public Tensor[] getRegressionTestData() {
    return TestUtils.loadSyntheticSupervisedData();
  }
  @Override
  public String getRegressionReferenceDirectory() {
    return getRegressionReferenceBaseDirectory() + "/supervised/synthetic/ref";
  }

}