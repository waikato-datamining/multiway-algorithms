package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;

import java.util.Map;

/**
 * Interface for algorithms that provide loading matrices after they are built.
 *
 * @author Steven Lang
 */
public interface LoadingMatrixAccessor {

  /**
   * Get a named map of loading matrices.
   *
   * @return Named map of loading matrices
   */
  Map<String, Tensor> getLoadingMatrices();
}
