package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.MultiBlockSupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.io.IOException;

/**
 * Regression test manager for MultiBlock supervised algorithms.
 *
 * @param <E> Algorithm type
 * @param <R> Output type
 * @author Steven Lang
 */
public abstract class MultiBlockSupervisedRegressionTestManager<E extends MultiBlockSupervisedAlgorithm, R> extends RegressionTestManager<E, R> {

  @Override
  public final boolean run() throws IOException {
    // Build algorithm input
    Tensor[] tensors = TestUtils.loadSyntheticMultiBlockSupervisedData();
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

  @Override
  public String getRegressionReferenceDirectory() {
    return getRegressionReferenceBaseDirectory() + "/supervised/synthetic/ref";
  }

}