package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.NTF.GRADIENT_UPDATE_TYPE;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.learning.config.Adam;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class NTFTest extends AbstractUnsupervisedAlgorithmTest<NTF> {

  NTF ntf;

  @Before
  public void init() {
    ntf = new NTF();
    ntf.setDebug(true);
    ntf.addStoppingCriterion(CriterionUtils.iterations(10));
  }

  @Test
  public void testCustomStepUpdater() {
    runWithUpdateType(GRADIENT_UPDATE_TYPE.STEP_UPDATE_CUSTOM);
  }

  @Test
  public void testCustomIterationUpdater() {
    runWithUpdateType(GRADIENT_UPDATE_TYPE.ITERATION_UPDATE_CUSTOM);
  }

  @Test
  public void testNormalizingUpdater() {
    runWithUpdateType(GRADIENT_UPDATE_TYPE.NORMALIZED_UPDATE);
  }

  private void runWithUpdateType(GRADIENT_UPDATE_TYPE iterationUpdateCustom) {
    int[] shape = {2, 3, 4, 5};
    Tensor X = TestUtils.generateRangeTensor(shape);
    ntf.setGradientUpdateType(iterationUpdateCustom);
    ntf.setUpdater(new Adam.Builder().learningRate(0.01).build());
    ntf.build(X);

    assertFalse(ntf.checkDecompositionForNegativeValues());
  }

  @Test
  public void testDecompositionShape() {
    int dimMode1 = 7;
    int dimMode2 = 4;
    int dimMode3 = 5;
    Tensor testData = TestUtils.generateRandomTensor(dimMode1, dimMode2, dimMode3);
    ntf.addStoppingCriterion(CriterionUtils.iterations(1));
    int numComponents = 6;
    ntf.setNumComponents(numComponents);
    ntf.build(testData);
    Tensor[] decomp = ntf.getDecomposition();
    int numModes = 3;
    assertEquals(numModes, decomp.length);
    assertEquals(dimMode1, decomp[0].size(0));
    assertEquals(numComponents, decomp[0].size(1));
    assertEquals(dimMode2, decomp[1].size(0));
    assertEquals(numComponents, decomp[1].size(1));
    assertEquals(dimMode3, decomp[2].size(0));
    assertEquals(numComponents, decomp[2].size(1));
  }

  @Override
  protected NTF constructAlgorithm() {
    return new NTF();
  }
}
