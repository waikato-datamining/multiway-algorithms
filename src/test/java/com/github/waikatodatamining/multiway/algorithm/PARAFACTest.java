package com.github.waikatodatamining.multiway.algorithm;

import com.github.waikatodatamining.multiway.TestUtils;
import com.github.waikatodatamining.multiway.algorithm.PARAFAC.Initialization;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

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

  private static final int maxIter = 1000;

  private PARAFAC pf;

  @Before
  public void init() {
    pf = new PARAFAC(F, numStarts, Initialization.RANDOM, maxIter);
  }

  @Test
  public void getLoadingMatrices() {
    pf.buildModel(TestUtils.generateRandomTensor(I, J, K));
    final double[][][] loadingMatrices = pf.getLoadingMatrices();
    assertEquals(3, loadingMatrices.length);
    assertEquals(I, loadingMatrices[0].length);
    assertEquals(J, loadingMatrices[1].length);
    assertEquals(K, loadingMatrices[2].length);
    assertEquals(F, loadingMatrices[0][0].length);
    assertEquals(F, loadingMatrices[1][0].length);
    assertEquals(F, loadingMatrices[2][0].length);
  }

  @Test
  public void getLossHistory() {
    pf.buildModel(TestUtils.generateRandomTensor(I, J, K));
    final List<List<Double>> lossHistory = pf.getLossHistory();

    assertEquals(3, lossHistory.size());
    lossHistory.forEach(h -> assertEquals(maxIter, h.size()));
  }
}