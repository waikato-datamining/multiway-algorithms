package com.github.waikatodatamining.multiway.algorithm;

import com.github.waikatodatamining.multiway.algorithm.stopping.CriterionType;
import com.github.waikatodatamining.multiway.algorithm.stopping.ImprovementStoppingCriterion;
import com.github.waikatodatamining.multiway.algorithm.stopping.StoppingCriterion;
import com.github.waikatodatamining.multiway.data.MathUtils;
import com.github.waikatodatamining.multiway.exceptions.InvalidInputException;
import com.google.common.collect.ImmutableSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Implementation of the PARAFAC algorithm according to
 * <a href='https://ac.els-cdn.com/S0169743997000324/1-s2.0-S0169743997000324-main.pdf?_tid=529932ca-e37c-11e7-b6f2-00000aab0f26&acdnat=1513551056_3f7d79d6c8151a442526fdfc89e28368'/>
 * <p>
 * Algorithm:
 * <ol>
 * <li>Initialize matrix B, C</li>
 * <li>Estimate A from X, B and C by least squares regression</li>
 * <li>Estimate B likewise</li>
 * <li>Estimate C likewise</li>
 * <li>Repeat until convergence</li>
 * </ol>
 * <p>
 *
 * @author Steven Lang
 */

public class PARAFAC extends AbstractAlgorithm {

  /** Number of components to reduce to */
  private int numComponents;

  /** Number of random starts */
  private int numStarts;

  /** Number of rows */
  private int numRows;

  /** Number of columns */
  private int numColumns;

  /** Number of dimensions */
  private int numDimensions;

  /** Array of size I x J x K */
  private INDArray X;

  /** Array of size I x F */
  private INDArray A;

  /** Array of size J x F */
  private INDArray B;

  /** Array of size K x F */
  private INDArray C;

  /** Loss history */
  private List<List<Double>> lossHist;

  /** Current loss */
  private double loss;


  /**
   * Constructor setting necessary options.
   *
   * @param numComponents Number of components
   * @param numStarts     Number of starts
   */
  public PARAFAC(int numComponents, int numStarts) {
    super();
    this.numComponents = numComponents;
    this.numStarts = numStarts;
    this.lossHist = new ArrayList<>();
  }

  /**
   * Input is assumed to be of the following shape: (I x J x K), where I is the
   * number of rows, J is the number of columns and K is the number of
   * measurements/components/slices
   *
   * @param input Input data
   */
  public void buildModel(double[][][] input) {
    validateInput(input);

    numRows = input.length;
    numColumns = input[0].length;
    numDimensions = input[0][0].length;

    X = MathUtils.from3dDoubleArray(input);

    for (int i = 0; i < numStarts; i++) {
      initComponentsRandom(i);

      // Collect loss for this run
      List<Double> losses = new ArrayList<>();

      while (!stoppingCriteriaMatches()) {
	// Run the nextIteration estimation iteration
	nextIteration();

	// Update algorithm state
	update();

	// Keep track of loss in this run
	losses.add(loss);
      }
      lossHist.add(losses);
      resetStoppingCriteria();
    }
  }

  /**
   * Initialize the component matrices with a random N(0,1) distribution
   *
   * @param seed Seed for the RNG
   */
  private void initComponentsRandom(int seed) {
    A = Nd4j.create(numRows, numComponents);
    B = Nd4j.randn(numColumns, numComponents, seed);
    C = Nd4j.randn(numDimensions, numComponents, seed + 1000);
  }


  /**
   * Execute the next iteration:
   * <p>
   * A = X*((C(+)B)^-1)^T
   * B = X*((C(+)A)^-1)^T
   * C = X*((B(+)A)^-1)^T
   * <p>
   * where (+) is the columnwise KhatriRao product.
   */
  private void nextIteration() {
    estimate(A, C, B, 0);
    estimate(B, C, A, 1);
    estimate(C, B, A, 2);
  }

  /**
   * Execute an estimation step for a specific component
   *
   * @param arrToUpdate The component which will be updated in this step
   * @param arr1        Left argument for kr-product
   * @param arr2        Right argument for kr-product
   * @param unfoldAxis  Indicate the axis at which the X tensor will be unfolded
   */
  private void estimate(INDArray arrToUpdate, INDArray arr1, INDArray arr2, int unfoldAxis) {
    final INDArray khatriRao = MathUtils.khatriRaoProductColumnWise(arr1, arr2);
    final INDArray inv = MathUtils.pseudoInvert(khatriRao, false);
    final INDArray transp = inv.transpose();
    final INDArray Xtmp = MathUtils.matricize(X, unfoldAxis);
    final INDArray tmp = Xtmp.mmul(transp);

    // Update array inplace
    arrToUpdate.assign(tmp);
  }

  /**
   * Get the decomposed loading matrices
   *
   * @return Loading matrices
   */
  public double[][][] getLoadingMatrices() {
    return new double[][][]{
      MathUtils.to2dDoubleArray(A),
      MathUtils.to2dDoubleArray(B),
      MathUtils.to2dDoubleArray(C)
    };
  }

  /**
   * Calculate the reconstruction calculateLoss
   *
   * @return Reconstruction calculateLoss
   */
  private double calculateLoss() {
    return MathUtils.matricize(X, 0).squaredDistance(reconstruct());
  }

  /**
   * Get the loss history over all reruns and iterations.
   * First index indicates the run and second index indicates the iteration.
   *
   * @return Loss history
   */
  public List<List<Double>> getLossHistory() {
    return lossHist;
  }

  /**
   * Reconstruct the matrix from the estimated components
   *
   * @return Reconstruction from the estimated components
   */
  private INDArray reconstruct() {
    return A.mmul(MathUtils.khatriRaoProductColumnWise(C, B).transpose());
  }

  /**
   * Validate the input data
   *
   * @param inputMatrix Input data
   */
  private void validateInput(double[][][] inputMatrix) {
    if (inputMatrix.length == 0
      || inputMatrix[0].length == 0
      || inputMatrix[0][0].length == 0) {
      throw new InvalidInputException("Input matrix dimensions must be " +
	"greater than 0.");
    }
  }


  @Override
  void update() {
    // Update loss
    loss = calculateLoss();

    // Update stopping criteria
    for (StoppingCriterion sc : stoppingCriteria) {
      switch (sc.getType()) {
	case ITERATION:
	  sc.update();
	  break;
	case TIME:
	  sc.update();
	  break;
	case IMPROVEMENT:
	  ((ImprovementStoppingCriterion) sc).update(loss);
	  break;
      }
    }
  }

  @Override
  Set<CriterionType> getAvailableStoppingCriteria() {
    return ImmutableSet.of(
      CriterionType.ITERATION,
      CriterionType.TIME,
      CriterionType.IMPROVEMENT
    );
  }
}
