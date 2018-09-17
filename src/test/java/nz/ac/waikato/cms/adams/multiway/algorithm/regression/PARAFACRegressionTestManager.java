package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.algorithm.PARAFAC;
import nz.ac.waikato.cms.adams.multiway.data.DataReader;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class PARAFACRegressionTestManager extends UnsupervisedRegressionTestManager<PARAFAC> {

  @Override
  public boolean resultEqualsReference() throws IOException {
    Map<String, Tensor> result = algorithm.getLoadingMatrices();
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
    Map<String, Tensor> decomp = algorithm.getLoadingMatrices();
    for (String s : decomp.keySet()){
      DataReader.writeMatrixCsv(decomp.get(s).toArray2d(), getReferenceFilePath(s), ",");
    }
  }

  @Override
  public Map<String, Tensor> loadReference() throws IOException {
    Map<String, Tensor> referenceDecomposition = new HashMap<>();
    for (String s : algorithm.getLoadingMatrices().keySet()){
      referenceDecomposition.put(s, Tensor.create(DataReader.readMatrixCsv(getReferenceFilePath(s),",")));
    }
    return referenceDecomposition;
  }
}