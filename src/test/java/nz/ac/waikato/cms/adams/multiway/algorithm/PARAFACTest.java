package nz.ac.waikato.cms.adams.multiway.algorithm;

import com.google.common.collect.ImmutableMap;
import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.PARAFAC.Initialization;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * {@link PARAFAC} algorithm testcase.
 *
 * @author Steven Lang
 */
public class PARAFACTest {

  private static final int F = 2;

  private static final int I = 5;

  private static final int J = 4;

  private static final int K = 3;

  private static final int numStarts = 3;

  private static final int maxIter = 10;

  private PARAFAC pf;

  @Before
  public void init() {
    pf = new PARAFAC();
    pf.setNumComponents(F);
    pf.setNumStarts(numStarts);
    pf.setInitMethod(Initialization.RANDOM);
    pf.addStoppingCriterion(CriterionUtils.iterations(maxIter));
  }

  @Test
  public void getLoadingMatrices() {

    pf.build(Tensor.create(TestUtils.generateRandomTensor(I, J, K)));
    final Map<String, Tensor> loadingMatrices = pf.getLoadingMatrices();
    assertEquals(3, loadingMatrices.size());
    assertEquals(I, loadingMatrices.get("A").size(0));
    assertEquals(J, loadingMatrices.get("B").size(0));
    assertEquals(K, loadingMatrices.get("C").size(0));
    assertEquals(2, loadingMatrices.get("A").order());
    assertEquals(2, loadingMatrices.get("B").order());
    assertEquals(2, loadingMatrices.get("C").order());
    assertEquals(F, loadingMatrices.get("A").size(1));
    assertEquals(F, loadingMatrices.get("B").size(1));
    assertEquals(F, loadingMatrices.get("C").size(1));
  }

  @Test
  public void getLossHistory() {
    pf.build(Tensor.create(TestUtils.generateRandomTensor(I, J, K)));
    final List<List<Double>> lossHistory = pf.getLossHistory();

    assertEquals(3, lossHistory.size());
    lossHistory.forEach(h -> assertEquals(maxIter, h.size()));
  }
}