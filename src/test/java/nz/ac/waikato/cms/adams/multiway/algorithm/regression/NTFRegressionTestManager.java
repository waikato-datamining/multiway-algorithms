package nz.ac.waikato.cms.adams.multiway.algorithm.regression;


import nz.ac.waikato.cms.adams.multiway.algorithm.NTF;
import nz.ac.waikato.cms.adams.multiway.data.DataReader;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import javax.management.StringValueExp;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class NTFRegressionTestManager extends UnsupervisedRegressionTestManager<NTF> {

  @Override
  public boolean resultEqualsReference() throws IOException {
    Tensor[] result = algorithm.getDecomposition();
    Map<String, Tensor> refMap = loadReference();


    for (int i = 0; i < result.length; i++) {
      if (!result[i].equalsWithEps(refMap.get(String.valueOf(i)), 10e-7)) {
	return false;
      }
    }

    return true;
  }

  @Override
  public void saveNewReference() throws IOException {
    Tensor[] decomp = algorithm.getDecomposition();
    for (int i = 0; i < decomp.length; i++) {
      DataReader.writeMatrixCsv(decomp[i].toArray2d(), getReferenceFilePath(String.valueOf(i)), ",");
    }
  }

  @Override
  public Map<String, Tensor> loadReference() throws IOException {
    Map<String, Tensor> ref = new HashMap<>();
    for (int i = 0; i < algorithm.getDecomposition().length; i++) {
      String path = getReferenceFilePath(String.valueOf(i));
      Tensor tensor = Tensor.create(DataReader.readMatrixCsv(path, ","));
      ref.put(String.valueOf(i), tensor);
    }
    return ref;
  }
}