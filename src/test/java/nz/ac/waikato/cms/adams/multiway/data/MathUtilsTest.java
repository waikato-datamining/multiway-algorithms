package nz.ac.waikato.cms.adams.multiway.data;

import nz.ac.waikato.cms.adams.multiway.TestUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static nz.ac.waikato.cms.adams.multiway.data.MathUtils.t;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Test cases for the static {@link MathUtils} methods.
 *
 * @author Steven Lang
 */
public class MathUtilsTest {

  /**
   * Example from: <a href="https://www.wolframalpha.com/input/?i=invert+matrix+((1,2),(3,4),(5,6))">here</a>
   */
  @Test
  public void testPseudoInvert() {
    INDArray X = Nd4j.create(new double[][]{{1, 2}, {3, 4}, {5, 6}});
    INDArray expected = Nd4j.create(new double[][]{{-16, -4, 8}, {13, 4, -5}}).mul(1 / 12d);
    INDArray actual = MathUtils.pseudoInvert2(X, false);
    assertEquals(expected, actual);

    final INDArray identity3x3 = Nd4j.create(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    final INDArray identity2x2 = Nd4j.create(new double[][]{{1, 0}, {0, 1}});
    final double precision = 1e-5;

    // right inverse
    final INDArray rightInverseCheck = X.mmul(actual);
    // right inverse must not hold since X rows are not linear independent (x_3 + x_1 = 2*x_2)
    assertFalse(rightInverseCheck.equalsWithEps(identity3x3, precision));

    // left inverse must hold since X columns are linear independent
    final INDArray leftInverseCheck = actual.mmul(X);
    assertTrue(leftInverseCheck.equalsWithEps(identity2x2, precision));

    // general condition
    final INDArray generalCond = X.mmul(actual).mmul(X);
    assertTrue(X.equalsWithEps(generalCond, precision));
  }

  /**
   * Testing the wikipedia example from <a href="https://en.wikipedia.org/wiki/Kronecker_product#Khatri%E2%80%93Rao_product">here</a>
   */
  @Test
  public void testKhatriRaoProductColumnWise() {
    INDArray C = Nd4j.create(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    INDArray D = C.transpose();

    INDArray expected = Nd4j.create(new double[][]{
      {1, 8, 21},
      {2, 10, 24},
      {3, 12, 27},
      {4, 20, 42},
      {8, 25, 48},
      {12, 30, 54},
      {7, 32, 63},
      {14, 40, 72},
      {21, 48, 81},
    });

    INDArray actual = MathUtils.khatriRaoProductColumnWise(C, D);
    assertEquals(expected, actual);


    /*
    Second example from R:
    ---
    x <- matrix(seq(20),4,5,byrow=TRUE)
    y <- matrix(c(21:40),4,5,byrow=TRUE)
    res <- KhatriRao(x,y)
    ---
    */
    final INDArray arange = Nd4j.arange(1, 21).reshape(4, 5);
    final INDArray arange2 = Nd4j.arange(21, 41).reshape(4, 5);
    final INDArray prod = MathUtils.khatriRaoProductColumnWise(arange, arange2);

    INDArray res = Nd4j.create(new double[][]{
      {21, 44, 69, 96, 125},
      {26, 54, 84, 116, 150},
      {31, 64, 99, 136, 175},
      {36, 74, 114, 156, 200},
      {126, 154, 184, 216, 250},
      {156, 189, 224, 261, 300},
      {186, 224, 264, 306, 350},
      {216, 259, 304, 351, 400},
      {231, 264, 299, 336, 375},
      {286, 324, 364, 406, 450},
      {341, 384, 429, 476, 525},
      {396, 444, 494, 546, 600},
      {336, 374, 414, 456, 500},
      {416, 459, 504, 551, 600},
      {496, 544, 594, 646, 700},
      {576, 629, 684, 741, 800}
    });

    assertEquals(prod, res);
  }

  @Test
  public void testInvertMatricize() {
    final INDArray X = Nd4j.arange(4 * 3 * 2).reshape(4, 3, 2);

    final INDArray matricized = MathUtils.matricize(X, 0);
    final INDArray folded = MathUtils.invertMatricize(matricized, 0, 3, 2);

    assertEquals(X, folded);
  }

  @Test
  public void testArrayConversion() {
    final int seed = 0;
    final int[] shape = {4, 3, 2};
    final INDArray X = Nd4j.randn(shape, seed);
    final INDArray rec = MathUtils.from3dDoubleArray(MathUtils.to3dDoubleArray(X));
    final INDArray rec2 = Tensor.create(Tensor.create(X).toArray3d()).getData();
    assertEquals(X, rec);
    assertEquals(X, rec2);
  }

  @Test
  public void testInvertVectorize() {
    final int I = 3;
    final int J = 4;
    final int K = 5;
    final INDArray X = TestUtils.generateRandomTensor(I, J, K).getData();
    final INDArray y = Nd4j.arange(0, I).transpose();

    // Use invertVectorize function
    final INDArray Xunfolded = MathUtils.matricize(X, 0);
    final INDArray Zactual = MathUtils.invertVectorize(Xunfolded.transpose().mmul(y), J);

    // Compute the expected Z matrix with z_j,k = sum_i (y_i*x_i,j,k)
    final INDArray Zexpected = Nd4j.create(J, K);
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
	double sum = 0;
	for (int i = 0; i < I; i++) {
	  sum += y.getDouble(i) * X.getDouble(i, j, k);
	}
	Zexpected.putScalar(j, k, sum);
      }
    }

    assertTrue(Zexpected.equalsWithEps(Zactual, 10E-5));
  }

  /**
   * Test Gram-Schmidt orthonormalization by checking that the dot product of
   * all column matrices in the new basis are zero and the column vectors are
   * of length one.
   */
  @Test
  public void testOrthogonalize() {

    final int numRows = 10;
    final int numCols = 50;
    final int seed = 42;

    final INDArray rand = Nd4j.rand(numRows, numCols, seed);
    for (boolean normalize : new boolean[]{true, false}) {
      final INDArray randOrth = MathUtils.orth(rand, normalize);
      for (int i = 0; i < rand.size(0); i++) {
	final INDArray ui = randOrth.getColumn(i);

	for (int j = 0; j < rand.size(0); j++) {
	  if (i == j) continue;
	  final INDArray uj = randOrth.getColumn(j);
	  final double prod = t(ui).mmul(uj).getDouble(0);
	  assertEquals(0, prod, 10e-5);

	  if (normalize) {
	    assertEquals(1.0, ui.norm2(0).getDouble(0), 10e-5);
	  }
	}
      }
    }
  }

  @Test
  public void testExtendArray(){
    long[] arr = {1,2,3};
    final int val = 4;
    long[] expectedArr = {1,2,3, val};

    long[] res = MathUtils.extendIdxToArray(arr, val);
    assertTrue(Arrays.equals(expectedArr, res));

    long[] expectedArrWithIdx = {1, val,2,3};
    int idx = 1;
    long[] resArrWithIdx = MathUtils.extendIdxToArray(arr, idx, val);
    assertTrue(Arrays.equals(expectedArrWithIdx, resArrWithIdx));
  }

  @Test
  public void testRemoveIdxFromArray(){
    long[] arr = {1,2,3,4,5};
    long[] expectedArr = {1,2,4,5};
    int removeIdx = 2;
    long[] res = MathUtils.removeIdxFromArray(arr, removeIdx);
    assertTrue(Arrays.equals(expectedArr, res));
  }

  @Test
  public void testOuter(){
    INDArray x = Nd4j.arange(1,4).reshape(3,1);
    INDArray y = Nd4j.zeros(2).reshape(2,1);
    y.putScalar(0, 2);
    double[][] expectedDoubleArray = {{2,0},{4,0},{6,0}};
    INDArray expected = Nd4j.create(expectedDoubleArray);
    final INDArray result = MathUtils.outer(x, y);
    assertTrue(result.equalsWithEps(expected, 10e-6));
  }

}