package data.tensor;

import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.data.tensor.TensorArrayFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jTensorArrayFactory implements TensorArrayFactory {

  public Tensor zeros(int... shape) {
    return tensor(Nd4j.zeros(shape));
  }

  public Tensor ones(int... shape) {
    return tensor(Nd4j.ones(shape));
  }

  public Tensor randn(int rows, int columns, long seed) {
    return tensor(Nd4j.randn(rows, columns, seed));
  }

  public Tensor randn(int dim1, int dim2, int dim3, long seed) {
    return tensor(Nd4j.randn(new int[]{dim1,dim2,dim3},seed));
  }

  public Tensor randn(int[] shape, long seed) {
    return tensor(Nd4j.randn(shape, seed));
  }

  private static Tensor tensor(INDArray toBeWrapped){
    return Nd4jTensor.create(toBeWrapped);
  }
}
