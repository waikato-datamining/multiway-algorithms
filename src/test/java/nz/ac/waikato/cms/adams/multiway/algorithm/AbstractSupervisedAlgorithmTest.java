package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.SupervisedAlgorithm;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

public abstract class AbstractSupervisedAlgorithmTest<T extends
  SupervisedAlgorithm> extends AbstractAlgorithmTest<T> {


  @Test
  public void testBuildWithNull() {
    assertNotNull(constructAlgorithm().build(null, null));
  }
}
