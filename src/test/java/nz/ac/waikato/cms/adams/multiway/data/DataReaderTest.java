package nz.ac.waikato.cms.adams.multiway.data;

import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertTrue;

/**
 * Testcases for {@link DataReader}.
 *
 * @author Steven Lang
 */
public class DataReaderTest {

  @Test
  public void read3WaySparseWithHeader() throws IOException {
    String path = "src/test/resources/datareader/threeway-test-data-with-header.csv";
    final double[][][] data = DataReader.read3WaySparse(path, ",", 3, true);

    double[][][] dataExpected = {
      {
	{0.0, 1.0, 2.0},
	{3.0, 4.0, 5.0},
	{6.0, 7.0, 8.0}
      },
      {
	{0.0, 1.0, 2.0},
	{3.0, 4.0, 5.0},
	{6.0, 7.0, 8.0}
      },
      {
	{0.0, 1.0, 2.0},
	{3.0, 4.0, 5.0},
	{6.0, 7.0, 8.0}
      }
    };

    assertTrue(Arrays.deepEquals(dataExpected, data));
  }

  @Test
  public void read3WaySparseWithoutHeader() throws IOException {
    String path = "src/test/resources/datareader/threeway-test-data-without-header.csv";
    final double[][][] data = DataReader.read3WaySparse(path, ",", 3, false);

    double[][][] dataExpected = {
      {
	{0.0, 1.0, 2.0},
	{3.0, 4.0, 5.0},
	{6.0, 7.0, 8.0}
      },
      {
	{0.0, 1.0, 2.0},
	{3.0, 4.0, 5.0},
	{6.0, 7.0, 8.0}
      },
      {
	{0.0, 1.0, 2.0},
	{3.0, 4.0, 5.0},
	{6.0, 7.0, 8.0}
      }
    };

    assertTrue(Arrays.deepEquals(dataExpected, data));
  }

  @Test
  public void read3WaySparseMissingValues() throws IOException {
    String path = "src/test/resources/datareader/threeway-test-data-missing-values.csv";
    final double[][][] data = DataReader.read3WaySparse(path, ",", 3, true);

    double[][][] dataExpected = {
      {
	{Double.NaN, 1.0, 2.0},
	{3.0, 4.0, 5.0},
	{6.0, 7.0, 8.0}
      },
      {
	{0.0, 1.0, 2.0},
	{3.0, Double.NaN, 5.0},
	{6.0, 7.0, 8.0}
      },
      {
	{0.0, 1.0, 2.0},
	{3.0, 4.0, 5.0},
	{6.0, 7.0, Double.NaN}
      }
    };

    assertTrue(Arrays.deepEquals(dataExpected, data));
  }
}