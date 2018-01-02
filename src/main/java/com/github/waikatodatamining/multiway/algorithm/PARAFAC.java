package com.github.waikatodatamining.multiway.algorithm;

import com.github.waikatodatamining.multiway.data.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

public class PARAFAC {

  /** Number of components to reduce to */
  private int numComponents;

  /** Maximum number of iterations */
  private int maxIter;

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
  private double[][] lossHist;

  public PARAFAC(int numComponents, int maxIter, int numStarts) {
    this.numComponents = numComponents;
    this.maxIter = maxIter;
    this.numStarts = numStarts;
    this.lossHist = new double[numStarts][maxIter];
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

    // Create array of slices
    X = Nd4j.create(numRows, numColumns, numDimensions);
    for (int i = 0; i < input.length; i++) {
      double[][] row = input[i];
      X.putRow(i, Nd4j.create(row));
    }


    for (int i = 0; i < numStarts; i++) {
      initComponentsRandom(i);

      for (int j = 0; j < maxIter; j++) {
	// Run the next estimation iteration
	next();

	// Collect loss
	lossHist[i][j] = loss();
      }
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
   * Execute the next iteration: Update
   */
  private void next() {
    estimate(A, B, C, 0);
    estimate(B, A, C, 1);
    estimate(C, A, B, 2);
  }

  /**
   * Execute an estimation step for a specific component
   *
   * @param arrToUpdate The component which will be updated in this step
   * @param arr1        Right argument for kr-product
   * @param arr2        Left argument for kr-product
   * @param unfoldAxis  Indicate the axis at which the X tensor will be unfolded
   */
  private void estimate(INDArray arrToUpdate, INDArray arr1, INDArray arr2, int unfoldAxis) {
    final INDArray khatriRao = MathUtils.khatriRaoProductColumnWise(arr2, arr1);
    final INDArray inv = MathUtils.pseudoInvert(khatriRao, false);
    final INDArray transp = inv.transpose();
    final INDArray Xtmp = MathUtils.matricize(X, unfoldAxis);
    final INDArray tmp = Xtmp.mmul(transp);

    // Update array inplace
    arrToUpdate.assign(tmp);
  }

  /**
   * Get the decomposed component
   *
   * @return Components
   */
  public double[][][] getComponents() {
    return new double[][][]{
      MathUtils.toDoubleMatrix(A),
      MathUtils.toDoubleMatrix(B),
      MathUtils.toDoubleMatrix(C)
    };
  }

  /**
   * Calculate the reconstruction loss
   *
   * @return Reconstruction loss
   */
  private double loss() {
    return MathUtils.matricize(X, 0).squaredDistance(reconstruct());
  }

  /**
   * Get the loss history over all reruns and iterations.
   * First index indicates the run and second index indicates the iteration.
   *
   * @return Loss history
   */
  public double[][] getLossHistory() {
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
    // TODO: do some validation on shape etc
  }
}
