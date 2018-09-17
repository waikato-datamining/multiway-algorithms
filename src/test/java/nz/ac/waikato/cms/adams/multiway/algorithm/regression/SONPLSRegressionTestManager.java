package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.algorithm.SONPLS;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * SONPLS regression test manager.
 *
 * @author Steven Lang
 */
public class SONPLSRegressionTestManager extends MultiBlockSupervisedRegressionTestManager<SONPLS> {

  /** Yhat specifier */
  private static final String YHAT = "Yhat";

  @Override
  public boolean resultEqualsReference() throws IOException {
    Map<String, Tensor> result = algorithm.getLoadingMatrices();
    Tensor[] data = getRegressionTestData();

    Tensor Yhat = algorithm.predict(new Tensor[]{data[0], data[1]});
    result.put(YHAT, Yhat);

    Map<String, Tensor> reference = loadReference();
    if (!result.keySet().equals(reference.keySet())){
      return false;
    }

    for (String key : result.keySet()){
      if (!result.get(key).equalsWithEps(reference.get(key), 10e-7)) {
	return false;
      }
    }
    return true;
  }

  @Override
  public void saveNewReference() throws IOException {
    Map<String, Tensor> ref = algorithm.getLoadingMatrices();

    for (String specifier : ref.keySet()){
      saveMatrix(specifier, ref.get(specifier));
    }

    Tensor[] data = getRegressionTestData();
    Tensor Yhat = algorithm.predict(new Tensor[]{data[0], data[1]});

    saveMatrix(YHAT, Yhat);
  }

  @Override
  public Map<String, Tensor> loadReference() throws IOException {
    Map<String, Tensor> ref = new HashMap<>();
    for (String specifier : algorithm.getLoadingMatrices().keySet()){
      ref.put(specifier, loadMatrix(specifier));
    }

    ref.put(YHAT, loadMatrix(YHAT));
    return ref;
  }
}
