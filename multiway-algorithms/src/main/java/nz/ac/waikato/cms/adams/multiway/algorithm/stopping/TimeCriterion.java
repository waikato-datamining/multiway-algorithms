package nz.ac.waikato.cms.adams.multiway.algorithm.stopping;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Criterion that checks if a certain amount of seconds has passed.
 *
 * @author Steven Lang
 */
public class TimeCriterion extends Criterion<Long> {

  /** Logger instance */
  private static final Logger log = LogManager.getLogger(TimeCriterion.class);

  /** Serial version UID */
  private static final long serialVersionUID = 1056852546265993835L;

  /** Stopwatch for time */
  protected StopWatch sw;

  /** Time elapsed */
  protected long secondsElapsed;

  /** Maximum time */
  protected long maxSeconds;

  /** Conversion of milliseconds to seconds */
  protected static final long MILLIS_TO_SECONDS = 1000L;

  @Override
  protected void initialize() {
    super.initialize();
    this.sw = new StopWatch();
  }

  /**
   * Get maximum runtime in seconds.
   *
   * @return Maximum runtime in seconds
   */
  public long getMaxSeconds() {
    return maxSeconds;
  }

  /**
   * Set maximum runtime in seconds.
   *
   * @param maxSeconds Maximum runtime in seconds
   */
  public void setMaxSeconds(long maxSeconds) {
    if (maxSeconds <= 0) {
      log.warn("Time criterion must be greater" +
	" than zero.");
    }
    else {
      this.maxSeconds = maxSeconds;
    }
  }

  @Override
  public boolean matches() {
    // Start stopwatch on first call
    if (!sw.isStarted()) {
      sw.start();
    }
    final boolean m = secondsElapsed >= maxSeconds;
    if (m) notify("Matched after: " + secondsElapsed + "s");
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
}
