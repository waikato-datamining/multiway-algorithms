package nz.ac.waikato.cms.adams.multiway.data;

import java.util.Objects;

/**
 * Straight forward immutable Tuple class with two protected members of variable
 * types.
 *
 * @param <A> Type of first element
 * @param <B> Type of second element
 * @author Steven Lang
 */
public class Tuple<A, B> {

  /** First element */
  protected A first;

  /** Second element */
  protected B second;

  /**
   * Tuple constructor.
   * @param first First element
   * @param second Second element
   */
  public Tuple(A first, B second) {
    this.first = first;
    this.second = second;
  }

  /**
   * Get the first element.
   * @return First element
   */
  public A getFirst() {
    return first;
  }

  /**
   * Get the second element.
   * @return Second element
   */
  public B getSecond() {
    return second;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Tuple<?, ?> tuple = (Tuple<?, ?>) o;
    return Objects.equals(first, tuple.first) &&
      Objects.equals(second, tuple.second);
  }

  @Override
  public int hashCode() {
    return Objects.hash(first, second);
  }

  @Override
  public String toString() {
    return "Tuple{" +
      "first=" + first +
      ", second=" + second +
      '}';
  }
}
