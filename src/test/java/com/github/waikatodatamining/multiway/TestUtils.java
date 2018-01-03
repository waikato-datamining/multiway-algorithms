package com.github.waikatodatamining.multiway;

import java.lang.reflect.Field;

/**
 * Test utilities.
 *
 * @author Steven Lang
 */
public class TestUtils {

  /**
   * Get a private field of a certain object by name.
   *
   * @param obj  Object to be accessed
   * @param name Fieldname
   * @param <T>  Class to return
   * @return Field value
   * @throws IllegalAccessException Access not allowed
   * @throws NoSuchFieldException   Field not found
   */
  public static <T> T getField(Object obj, String name) throws IllegalAccessException, NoSuchFieldException {
    Field field = obj.getClass().getDeclaredField(name);
    field.setAccessible(true);
    return (T) field.get(obj);
  }
}
