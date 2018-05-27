package nz.ac.waikato.cms.adams.multiway.algorithm.regression;


import nz.ac.waikato.cms.adams.multiway.algorithm.NTF;
import nz.ac.waikato.cms.adams.multiway.data.DataReader;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.io.IOException;

public class NTFRegressionTestManager extends UnsupervisedRegressionTestManager<NTF, Tensor[]> {

  @Override
  public boolean resultEqualsReference() throws IOException {
    Tensor[] result = algorithm.getDecomposition();
    Tensor[] reference = loadReference();

    for (int i = 0; i < result.length; i++) {
      if (!result[i].equalsWithEps(reference[i], 10e-7)) {
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
  public Tensor[] loadReference() throws IOException {
    Tensor[] referenceDecomposition = new Tensor[algorithm.getDecomposition().length];
    for (int i = 0; i < referenceDecomposition.length; i++) {
      String path = getReferenceFilePath(String.valueOf(i));
      referenceDecomposition[i] = Tensor.create(DataReader.readMatrixCsv(path, ","));
    }
    return referenceDecomposition;
  }

  @Override
  public String getRegressionReferenceDirectory() {
    return super.getRegressionReferenceDirectory() + "/ntf/" + options;
  }

}