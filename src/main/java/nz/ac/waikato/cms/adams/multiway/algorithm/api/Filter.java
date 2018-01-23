package nz.ac.waikato.cms.adams.multiway.algorithm.api;

import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

/**
 * Filter interface that provides a filter method
 */
public interface Filter {

  /**
   * Apply the filter to new input data and return the transformed input.
   *
   * @param input Input data
   * @return Transformed data
   */
  Tensor filter(Tensor input);
}
