package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.NTF.GRADIENT_UPDATE_TYPE;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class NTFTest extends AbstractUnsupervisedAlgorithmTest<NTF> {

  NTF ntf;

  @Before
  public void init() {
    ntf = new NTF();
    ntf.setDebug(true);
    ntf.addStoppingCriterion(CriterionUtils.iterations(1000));
  }

  @Test
  public void testBuild() {
    int[] shape = {2, 3, 4, 5};
    int range = Arrays.stream(shape).reduce(1, (l, r) -> l * r);
    final Tensor X = Tensor.create(Nd4j.arange(range).reshape(shape));
    final int numComponents = 2;
    ntf.setNumComponents(numComponents);
    ntf.addStoppingCriterion(CriterionUtils.iterations(10000));
    ntf.setGradientUpdateType(GRADIENT_UPDATE_TYPE.STEP_UPDATE_CUSTOM);
    ntf.setUpdater(new Adam.Builder().learningRate(0.001).build());
    ntf.build(X);
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
