package nz.ac.waikato.cms.adams.multiway.algorithm.regression;


import nz.ac.waikato.cms.adams.multiway.algorithm.NTF;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Map;

/**
 * NTF regression test manager.
 *
 * @author Steven Lang
 */
public class NTFRegressionTestManager extends UnsupervisedRegressionTestManager<NTF> {

  @Override
  protected Map<String, Tensor> generateResults() {
    Map<String, Tensor> references = super.generateResults();
    Tensor[] decomp = algorithm.getDecomposition();
    for (int i = 0; i < decomp.length; i++) {
      references.put("decomp-" + i, decomp[i]);
    }
    return references;
  }

  @Override
  public Tensor[] getRegressionTestData() {
    // Take abs value of all test data since NTF only operates on positive data
    Tensor[] regressionTestData = super.getRegressionTestData();
    for (int i = 0; i < regressionTestData.length; i++) {
      Tensor datum = regressionTestData[i];
      INDArray abs = Transforms.abs(datum.getData());
      regressionTestData[i] = Tensor.create(abs);
    }

    return regressionTestData;
  }
}