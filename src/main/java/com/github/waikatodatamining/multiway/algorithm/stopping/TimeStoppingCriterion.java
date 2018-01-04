package com.github.waikatodatamining.multiway.algorithm.stopping;

import com.github.waikatodatamining.multiway.exceptions.InvalidInputException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.time.StopWatch;

/**
 * Stopping criterion that checks if a certain amount of seconds has passed.
 *
 * @author Steven Lang
 */
@Slf4j
public class TimeStoppingCriterion extends StoppingCriterion<Long> {

  /** Stopwatch for time*/
  private StopWatch sw;

  /** Time elapsed */
  private long secondsElapsed;

  /** Maximum time */
  private long maxSeconds;

  /** Conversion of milliseconds to seconds */
  private static final long MILLIS_TO_SECONDS = 1000L;

  /**
   * Construct time criterion with given number of maximum seconds.
   *
   * @param maxSeconds Maximum number of seconds to elapse
   */
  public TimeStoppingCriterion(long maxSeconds) {
    this.maxSeconds = maxSeconds;
    this.sw = new StopWatch();
    validateParameters();
  }

  @Override
  public boolean matches() {
    // Start stopwatch on first call
    if (!sw.isStarted()) {
      sw.start();
    }
    final boolean m = secondsElapsed >= maxSeconds;
    if (m) notify("Stopped after: " + secondsElapsed + "s");
    return m;
  }

  @Override
  public void update() {
    secondsElapsed = sw.getTime() / MILLIS_TO_SECONDS;
  }

  @Override
  public CriterionType getType() {
    return CriterionType.TIME;
  }

  @Override
  public void reset() {
    sw.reset();
  }

  @Override
  public void validateParameters() {
    if (maxSeconds <= 0){
      throw new InvalidInputException("Time criterion must be greater" +
        " than zero.");
    }
  }
}
