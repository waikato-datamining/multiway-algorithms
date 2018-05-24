package nz.ac.waikato.cms.adams.multiway.algorithm;

import com.google.common.collect.ImmutableSet;
import nz.ac.waikato.cms.adams.multiway.algorithm.api.UnsupervisedAlgorithm;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.Criterion;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionType;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.CriterionUtils;
import nz.ac.waikato.cms.adams.multiway.algorithm.stopping.ImprovementCriterion;
import nz.ac.waikato.cms.adams.multiway.data.MathUtils;
import nz.ac.waikato.cms.adams.multiway.data.tensor.Tensor;
import nz.ac.waikato.cms.adams.multiway.exceptions.InvalidInputException;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.conditions.LessThan;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Non-Negative Tensor Factorization.
 *
 * <p>Implementation according to: <a href="https://dl.acm.org/citation.cfm?id=1102451">Non-negative
 * tensor factorization with applications to statistics and computer vision</a>
 */
public class NTF extends UnsupervisedAlgorithm {

  private static final long serialVersionUID = -3637653705671885273L;

  /** The number of modes of the input tensor (variable n in the paper) */
  protected int numModes;

  /** Number of components in the decomposed matrices (variable k in the paper) */
  protected int numComponents;

  /** Matrix array containing the decomposed matrices */
  protected INDArray[] decomposition;

  protected GradientManager gradientManager;

  /** Input tensor (variable G in the paper */
  protected INDArray target;

  /** Logger */
  protected static Logger logger = LoggerFactory.getLogger(NTF.class);

  protected static double EPS = 10e-8;

  protected double loss = Double.MAX_VALUE;

  protected List<Double> lossHistory;

  protected IUpdater updater;

  protected GRADIENT_UPDATE_TYPE gradientUpdateType;

  public void setUpdater(IUpdater updater) {
    if (GRADIENT_UPDATE_TYPE.NORMALIZED_UPDATE.equals(gradientUpdateType)) {
      logger.warn(
	"Setting an updater has no effect when the option "
	  + "gradient update type is set to {}.", GRADIENT_UPDATE_TYPE.NORMALIZED_UPDATE);
    }
    this.updater = updater;
  }

  @Override
  protected void initialize() {
    super.initialize();
    setUpdater(new Sgd(0.01));
    gradientUpdateType = GRADIENT_UPDATE_TYPE.NORMALIZED_UPDATE;
    lossHistory = new ArrayList<>();
    numComponents = 3;
    addStoppingCriterion(CriterionUtils.iterations(1000));
  }

  @Override
  protected String doBuild(Tensor x) {
    target = x.getData();
    numModes = target.shape().length;
    initializeGradientManager();
    initializeDecompositionMatrices();

    while (!stoppingCriteriaMatch()) {
      updateDecompositionStep();
      lossHistory.add(loss);

      if (isDebug) {
	logger.debug("Loss={}", getLoss());
      }
    }

    return null;
  }

  protected void initializeGradientManager() {
    switch (gradientUpdateType) {
      case NORMALIZED_UPDATE:
	gradientManager = null;
	break;
      case STEP_UPDATE_CUSTOM:
	gradientManager = new StepGradientManager(updater,
	  target.shape(), numComponents);
	break;
      case ITERATION_UPDATE_CUSTOM:
	gradientManager = new IterationGradientManager(updater, numModes,
	  numComponents);
	break;
      default:
	throw new InvalidInputException("Undefined gradient update type: " + gradientUpdateType);
    }
  }

  /**
   * Check the decomposition matrices for negative values. Mainly for debugging purposes.
   *
   * @return True if decomposition contains negative values.
   */
  protected boolean checkDecompositionForNegativeValues() {
    for (INDArray u : decomposition) {
      if (Transforms.sign(u).cond(new LessThan(0)).sumNumber().intValue() > 0) {
	return true;
      }
    }
    return false;
  }

  /** Update the internal state. */
  protected void updateStoppingCriteria() {
    // Update loss
    loss = getLoss();

    if (isDebug && checkDecompositionForNegativeValues()) {
      logger.error("The decomposition contains negative values, this should not occur.");
    }

    // Update stopping criteria states
    for (Criterion sc : stoppingCriteria.values()) {
      switch (sc.getType()) {
	case IMPROVEMENT:
	  ((ImprovementCriterion) sc).update(loss);
	  break;
	default:
	  sc.update();
      }
    }
  }

  /** A single update step. Updates each decomposition value once. */
  protected void updateDecompositionStep() {
    for (int mode = 0; mode < numModes; mode++) {
      for (int component = 0; component < numComponents; component++) {
	final int dimModeR = target.size(mode);
	for (int dimensionIdx = 0; dimensionIdx < dimModeR; dimensionIdx++) {
	  if (GRADIENT_UPDATE_TYPE.NORMALIZED_UPDATE.equals(gradientUpdateType)) {
	    updateSingleDecompositionValue(mode, component, dimensionIdx);
	  }
	  else {
	    updateSingleDecompositionValueCustomUpdater(mode, component, dimensionIdx);
	  }
	}
      }
    }


    if (GRADIENT_UPDATE_TYPE.ITERATION_UPDATE_CUSTOM.equals(gradientUpdateType)) {
      gradientManager.applyUpdate();
    }

    updateStoppingCriteria();
  }

  /**
   * Update a single variable as described in the paper.
   *
   * @param mode         Paper variable: r
   * @param component    Paper variable: s
   * @param dimensionIdx Paper variable l
   */
  private void updateSingleDecompositionValue(
    final int mode, final int component, final int dimensionIdx) {
    double nominator = getUpdateRuleNominator(mode, component, dimensionIdx);
    double denominator = getUpdateRuleDenominator(mode, component, dimensionIdx);
    final double uValueOld = getDecompositionValue(mode, dimensionIdx, component);
    double uValueNew = uValueOld * (nominator / (denominator + EPS));

    // Update u
    putSingleDecompositionValue(mode, dimensionIdx, component, uValueNew);
  }

  /**
   * Update a single variable as described in the paper, with the modification of a learning rate.
   *
   * @param mode         Paper variable: r
   * @param component    Paper variable: s
   * @param dimensionIdx Paper variable l
   */
  protected void updateSingleDecompositionValueCustomUpdater(
    final int mode, final int component, final int dimensionIdx) {

    double nominator = getUpdateRuleNominator(mode, component, dimensionIdx);
    double denominator = getUpdateRuleDenominator(mode, component, dimensionIdx);
    final double gradient = denominator - nominator;

    gradientManager.putGradient(mode, dimensionIdx, component, gradient);
  }

  /**
   * Get the nominator in the update rule specified in the paper (chapter 4).
   *
   * @param mode      Mode index
   * @param component Component index
   * @param dimension Dimension index
   * @return Update rule nominator.
   */
  protected double getUpdateRuleNominator(
    final int mode, final int component, final int dimension) {
    final int[] gShapeWithoutCurrentMode = getTargetShapeWithoutMode(mode);
    NdIndexIterator shapeIndexIterator = new NdIndexIterator(gShapeWithoutCurrentMode);
    double sum = 0;
    double maxIterations = Arrays.stream(gShapeWithoutCurrentMode).reduce(1, (prod, i) -> prod * i);
    double currentIteration = 0;
    while (shapeIndexIterator.hasNext() && currentIteration++ < maxIterations) {
      final int[] currentIndices = shapeIndexIterator.next();
      sum += getNextIndexStepInNominatorSum(mode, component, dimension, currentIndices);
    }
    return sum;
  }

  /**
   * Calculate a step in the nominator sum (see update rule, paper chapter 4)
   *
   * @param mode      Mode index
   * @param component Component index
   * @param dimension Dimension index
   * @param indices   Indices that point to the target value.
   * @return Nominator sum step result.
   */
  private double getNextIndexStepInNominatorSum(
    int mode, int component, int dimension, int[] indices) {
    final int[] currentIndicesExpanded = MathUtils.extendIdxToArray(indices, mode, dimension);
    double product = getUpdateRuleNominatorProduct(mode, component, currentIndicesExpanded);
    final double gValueAtCurrentIndices = target.getDouble(currentIndicesExpanded);
    return gValueAtCurrentIndices * product;
  }

  /**
   * Get the product in the update rule nominator specified in the paper (chapter 4).
   *
   * @param mode      Mode index
   * @param component Component index
   * @param indices   Set of indices that point to the correct decomposition element
   * @return Update rule denominator.
   */
  private double getUpdateRuleNominatorProduct(int mode, int component, int[] indices) {
    double product = 1;
    for (int currentMode = 0; currentMode < numModes; currentMode++) {
      // Skip mode m
      if (currentMode == mode) {
	continue;
      }
      final int row = indices[currentMode];
      double uValue = getDecompositionValue(currentMode, row, component);
      product = product * uValue;
    }
    return product;
  }

  /**
   * Get the denominator in the update rule specified in the paper (chapter 4).
   *
   * @param mode      Mode index
   * @param component Component index
   * @param dimension Dimension index
   * @return Update rule denominator.
   */
  protected double getUpdateRuleDenominator(
    final int mode, final int component, final int dimension) {
    double sum = 0;
    for (int currentComponent = 0; currentComponent < numComponents; currentComponent++) {
      double product = 1.0;
      for (int currentMode = 0; currentMode < numModes; currentMode++) {
	if (currentMode == mode) {
	  continue;
	}
	final INDArray uColJ = getDecompositionComponent(currentMode, currentComponent);
	final INDArray uColS = getDecompositionComponent(currentMode, component);
	final double uProdResult = uColJ.transpose().mmul(uColS).getDouble(0);
	product = product * uProdResult;
      }

      sum = sum + getDecompositionValue(mode, dimension, currentComponent) * product;
    }
    return sum;
  }

  /**
   * Get a decomposition component of a certain mode.
   *
   * @param mode      Mode index
   * @param component Component index
   * @return Decomposition value at the given index.
   */
  protected INDArray getDecompositionComponent(int mode, int component) {
    return decomposition[mode].getColumn(component);
  }

  /**
   * Get the decomposition of at the given index.
   *
   * @param mode      Mode index
   * @param row       Row index
   * @param component Component index
   * @return Decomposition value at the given index.
   */
  protected double getDecompositionValue(final int mode, final int row, final int component) {
    return decomposition[mode].getDouble(row, component);
  }

  /**
   * Put a single value into the decomposition at the given indices.
   *
   * @param mode      Mode index
   * @param row       Row index
   * @param component Component index
   * @param value     Value to put at the specific index
   */
  protected void putSingleDecompositionValue(
    final int mode, final int row, final int component, final double value) {
    decomposition[mode].putScalar(row, component, value);
  }

  /**
   * Construct the shape that is equal to the target's shape but with the given mode removed.
   *
   * @param mode Mode to leave out
   * @return Target shape with the given mode removed.
   */
  protected int[] getTargetShapeWithoutMode(final int mode) {
    final int[] gShape = target.shape();
    return MathUtils.removeIdxFromArray(gShape, mode);
  }

  /**
   * Initialize #numModes matrices where the matrix at index i is of shape (d_i x numComponents) and
   * d_i is the dimension of the i-th mode.
   */
  protected void initializeDecompositionMatrices() {
    final int seed = 0;
    decomposition = new INDArray[numModes];
    for (int i = 0; i < numModes; i++) {
      final int dimModeI = target.size(i);
      decomposition[i] = Transforms.abs(Nd4j.randn(dimModeI, numComponents, seed + i * 1000));
    }
  }

  protected double getLoss() {
    return target.squaredDistance(getReconstruction());
  }

  /**
   * Reconstruct the target matrix from the decomposition. This computes the outer product over
   * decomposition vector in each mode, which gives a rank-1 tensor for each mode. This is done for
   * each component and finally summed up.
   *
   * @return Target reconstruction
   */
  protected INDArray getReconstruction() {
    INDArray Greconstructed = Nd4j.zeros(target.shape());
    for (int k = 0; k < numComponents; k++) {
      // Rebuild each rank-1 tensor by taking the outer products over all modes
      INDArray ccCurrentModeComponent = getDecompositionComponent(0, k);
      for (int i = 1; i < numModes; i++) {
	INDArray ccNextModeComponent = getDecompositionComponent(i, k);
	ccCurrentModeComponent =
	  MathUtils.outer(ccCurrentModeComponent.dup(), ccNextModeComponent.dup());
      }

      Greconstructed.addi(ccCurrentModeComponent);
    }

    return Greconstructed;
  }

  @Override
  protected void resetState() {
    super.resetState();
    decomposition = null;
    numModes = 0;
    numComponents = 0;
  }

  public int getNumComponents() {
    return numComponents;
  }

  public void setNumComponents(int numComponents) {
    if (numComponents < 1) {
      throw new InvalidInputException(
	"Number of components needs to be at " + "least one or higher");
    }
    this.numComponents = numComponents;
  }

  @Override
  protected Set<CriterionType> getAvailableStoppingCriteria() {
    return ImmutableSet.of(CriterionType.IMPROVEMENT, CriterionType.ITERATION, CriterionType.TIME);
  }

  public GRADIENT_UPDATE_TYPE getGradientUpdateType() {
    return gradientUpdateType;
  }

  public void setGradientUpdateType(GRADIENT_UPDATE_TYPE gradientUpdateType) {
    this.gradientUpdateType = gradientUpdateType;
  }

  public Tensor[] getDecomposition() {
    Tensor[] decomp = new Tensor[numModes];
    for (int mode = 0; mode < numModes; mode++) {
      decomp[mode] = Tensor.create(decomposition[mode]);
    }
    return decomp;
  }

  /**
   * Enum defining the type of updates that shall be applied to construct the
   * decomposition values.
   */
  public enum GRADIENT_UPDATE_TYPE {
    /**
     * Default implementation as described in the paper, based a gradient
     * normalizing learning rate.
     */
    NORMALIZED_UPDATE,
    /**
     * Custom optimization strategy update after the calculation of each
     * single value in the decomposition.
     */
    STEP_UPDATE_CUSTOM,
    /**
     * Custom optimization strategy update after the calculation of all
     * values in the decomposition (after one iteration).
     */
    ITERATION_UPDATE_CUSTOM
  }

  /** A class that handles gradient updates for a specific set of gradients. */
  private class GradientWrapper implements Serializable {

    private static final long serialVersionUID = 6671164637702158204L;

    private GradientUpdater updater;

    private INDArray gradients;

    private int iteration = 0;

    private GradientWrapper(IUpdater updater, int rows, int numComponents) {
      this.gradients = Nd4j.create(rows, numComponents);
      final long stateSize = updater.stateSize(gradients.length());
      // Handle updater without states
      INDArray updateParamViewArray = null;
      if (stateSize > 0) {
	updateParamViewArray = Nd4j.zeros(1, (int) stateSize);
      }
      this.updater = updater.instantiate(updateParamViewArray, true);
    }

    /**
     * Put a gradient at the specified index in the gradient matrix.
     *
     * @param row       Row index
     * @param component Component index
     * @param gradient  Gradient value
     */
    private void putGradient(int row, int component, double gradient) {
      gradients.putScalar(row, component, gradient);
    }

    /**
     * Apply the gradient update, given the update rules in the underlying updater.
     */
    private void applyUpdate() {
      final INDArray gradientsFlattened = Nd4j.toFlattened(gradients);
      updater.applyUpdater(gradientsFlattened, iteration);
      gradients = gradientsFlattened.reshape(gradients.shape());
      iteration++;
    }
  }

  private abstract class GradientManager implements Serializable {

    private static final long serialVersionUID = -4450675200897774455L;


    /**
     * Put a gradient at the specified index in the gradient matrix.
     *
     * @param row       Row index
     * @param component Component index
     * @param gradient  Gradient value
     * @Override protected boolean isBatchUpdater() {
     * return false;
     * }
     */
    protected abstract void putGradient(int mode, int row, int component,
					double gradient);

    /**
     * Apply the gradient updates for each GradientWrapper, given the update rules in the underlying
     * updater.
     */
    protected abstract void applyUpdate();

  }


  /**
   * Manager class, that handles all gradient matrices, updaters und
   * gradient updates batch-wise for each mode. Updates occur after each
   * step (calculating the gradients for each single decomposition value
   * once.
   */
  private class IterationGradientManager extends GradientManager implements Serializable {

    private static final long serialVersionUID = -3147647560296068748L;

    /** One gradient wrapper for each mode */
    protected GradientWrapper[] gradientWrappers;

    private IterationGradientManager(IUpdater updater, int numModes,
				     int numComponents) {
      gradientWrappers = new GradientWrapper[numModes];
      for (int i = 0; i < numModes; i++) {
	int dimModeI = target.size(i);
	gradientWrappers[i] = new GradientWrapper(updater.clone(), dimModeI, numComponents);
      }
    }

    protected void putGradient(int mode, int row, int component, double gradient) {
      gradientWrappers[mode].putGradient(row, component, gradient);
    }

    protected void applyUpdate() {
      for (int mode = 0; mode < gradientWrappers.length; mode++) {
	GradientWrapper gw = gradientWrappers[mode];
	gw.applyUpdate();
	decomposition[mode].subi(gradientWrappers[mode].gradients);

	// Project decomposition back into the domain-space by clipping the
	// decomposition values
	BooleanIndexing.replaceWhere(decomposition[mode], 0,
	  Conditions.lessThan(0));
      }

    }
  }

  /**
   * Manager class, that handles all gradient matrices, updaters und
   * gradient updates batch-wise for each mode. Updates occur after each
   * iteration (calculating the gradients for each single decomposition value
   * once.
   */
  private class StepGradientManager extends GradientManager implements Serializable {

    private static final long serialVersionUID = 6247999674170790763L;

    /** One gradient wrapper for each mode */
    protected GradientWrapper[][][] gradientWrappers;

    private StepGradientManager(IUpdater updater, int[] targetShape,
				int numComponents) {
      gradientWrappers = new GradientWrapper[targetShape.length][0][0];
      for (int mode = 0; mode < targetShape.length; mode++) {
	int modeDim = targetShape[mode];
	gradientWrappers[mode] = new GradientWrapper[modeDim][numComponents];
	for (int row = 0; row < modeDim; row++) {
	  for (int component = 0; component < numComponents; component++) {

	    GradientWrapper gw = new GradientWrapper(updater.clone(), 1, 1);
	    gradientWrappers[mode][row][component] = gw;
	  }
	}
      }
    }

    protected void putGradient(int mode, int row, int component, double gradient) {
      gradientWrappers[mode][row][component].putGradient(0, 0, gradient);
      gradientWrappers[mode][row][component].applyUpdate();
      updateDecomposition(mode, row, component,
	gradientWrappers[mode][row][component].gradients.getDouble(0));
    }

    protected void updateDecomposition(int mode, int row, int component,
				       double gradient) {
      double uOld = decomposition[mode].getScalar(row, component).getDouble(0);
      double uNew = uOld - gradient;
      uNew = Math.max(0, uNew); // Clip
      decomposition[mode].putScalar(row, component, uNew);
    }

    protected void applyUpdate() {
      // Skip, as updates are applied while updating the gradient in
      // `putGradient`
    }
  }
}
