package nz.ac.waikato.cms.adams.multiway.algorithm;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
/**
 * {@link MultiLinearPLS} algorithm testcase.
 *
 * @author Steven Lang
 */
public class MultiLinearPLSTest {

  @Test
  public void doBuild() {
    final Tensor X = TestUtils.generateRandomTensor(10, 5, 3);
    final Tensor Y = TestUtils.generateRandomMatrix(10, 2);
    MultiLinearPLS mpls = new MultiLinearPLS();
    mpls.addStoppingCriterion(CriterionUtils.iterations(10));
    final String err = mpls.build(X, Y);
    assertTrue("Error was: " + err, err == null);
  }

  @Test
  public void filter() {
    final int numTestRows = 55;
    final int F = 9;
    final Tensor X = TestUtils.generateRandomTensor(10, 5, 3);
    final Tensor X2 = TestUtils.generateRandomTensor(numTestRows, 5, 3);
    final Tensor Y = TestUtils.generateRandomMatrix(10, 2);
    MultiLinearPLS mpls = new MultiLinearPLS();
    mpls.setNumComponents(F);
    mpls.addStoppingCriterion(CriterionUtils.iterations(10));
    mpls.build(X, Y);
    final Tensor filter = mpls.filter(X2);


    assertEquals(2, filter.order());
    assertEquals(numTestRows, filter.size(0));
    assertEquals(F, filter.size(1));
  }
}