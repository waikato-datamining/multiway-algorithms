package nz.ac.waikato.cms.adams.multiway.algorithm.regression;

import nz.ac.waikato.cms.adams.multiway.algorithm.api.SupervisedAlgorithm;

public abstract class SupervisedRegressionTestManager<E extends SupervisedAlgorithm, R> extends RegressionTestManager<E, R> {

  @Override
  public final boolean run() {
    return false;
  }
}