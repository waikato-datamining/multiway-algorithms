package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.AbstractAlgorithm;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;

public abstract class AbstractAlgorithmTest<T extends AbstractAlgorithm> {

  protected abstract T constructAlgorithm();

  @Test
  public abstract void testBuildWithNull();
}
