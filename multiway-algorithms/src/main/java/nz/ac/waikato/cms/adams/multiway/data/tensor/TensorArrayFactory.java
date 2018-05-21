package nz.ac.waikato.cms.adams.multiway.data.tensor;

import nz.ac.waikato.cms.locator.ClassLister;

import java.util.Properties;

public interface TensorArrayFactory {
    Tensor zeros(int... shape);
    Tensor ones(int... shape);
    Tensor randn(int rows, int columns, long seed);
    Tensor randn(int dim1, int dim2, int dim3, long seed);
    Tensor randn(int[] shape, long seed);
}
