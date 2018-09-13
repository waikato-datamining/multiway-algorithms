package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.MultiBlockSupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public abstract class MultiBlockSupervisedRegressionTestManager<E extends MultiBlockSupervisedAlgorithm, R> extends RegressionTestManager<E, R> {

  @Override
  public final boolean run() throws IOException {
    Tensor[] tensors = getRegressionTestData();
    Tensor[] xblocks = new Tensor[]{tensors[0], tensors[1]};
    Tensor y = tensors[2];
    algorithm.build(xblocks, y);
    if (checkIfReferenceExists()) {
      return resultEqualsReference();
    }
    else {
      saveNewReference();
      return true;
    }
  }

  public Tensor[] getRegressionTestData() {
    return TestUtils.loadSyntheticMultiBlockSupervisedData();
  }

  @Override
  public String getRegressionReferenceDirectory() {
    return getRegressionReferenceBaseDirectory() + "/supervised/synthetic/ref";
  }

}