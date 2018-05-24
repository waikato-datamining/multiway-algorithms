package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.SupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.UnsupervisedAlgorithm;
import org.junit.Assert;

import static org.junit.Assert.assertNotNull;

public abstract class AbstractUnsupervisedAlgorithmTest<T extends
  UnsupervisedAlgorithm> extends AbstractAlgorithmTest<T> {

  @Override
  public void testBuildWithNull() {
    assertNotNull(constructAlgorithm().build(null));
  }
}
