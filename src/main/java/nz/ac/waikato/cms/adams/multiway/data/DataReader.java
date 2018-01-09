package nz.ac.waikato.cms.adams.multiway.data;

import org.apache.commons.lang3.ArrayUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Data reader class.
 *
 * @author Steven Lang
 */
public class DataReader {

  /**
   * Read three-way sparse data of the following format:
   * x0 y0 z0 value0
   * x0 y0 z1 value1
   * ...
   *
   * Missing points default to {@link Double#NaN}.
   *
   * @param path      Path to data
   * @param sep       CSV separator
   * @param valueIdx  Data value index
   * @param hasHeader File includes a header at the top
   * @return Data array
   * @throws IOException Could not access given path
   */
  public static double[][][] read3WaySparse(String path,
					    String sep,
					    int valueIdx,
					    boolean hasHeader)
    throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(path))) {
      String line;
      int rowCount = 0;
      List<Integer> x = new ArrayList<>();
      List<Integer> y = new ArrayList<>();
      List<Integer> z = new ArrayList<>();
      List<Double> vals = new ArrayList<>();

      int[] idxs = new int[]{0, 1, 2, 3};
      ArrayUtils.removeElement(idxs, valueIdx);

      while ((line = br.readLine()) != null) {

        // Skip header if present
	if (hasHeader && rowCount == 0) {
	  rowCount++;
	  continue;
	}
	else if (line.isEmpty()) {
	  continue;
	}

	final String[] split = line.split(sep);
	rowCount++;

	// Add indices and corresponding value
	x.add(Integer.valueOf(split[idxs[0]]));
	y.add(Integer.valueOf(split[idxs[1]]));
	z.add(Integer.valueOf(split[idxs[2]]));
	vals.add(Double.valueOf(split[valueIdx]));
      }

      // Get maximum indices
      int maxX = Collections.max(x);
      int maxY = Collections.max(y);
      int maxZ = Collections.max(z);

      // Init NaN data matrix
      double[][][] data = new double[maxX + 1][maxY + 1][maxZ + 1];
      for (int i = 0; i < maxX + 1; i++) {
	for (int j = 0; j < maxY + 1; j++) {
	  for (int k = 0; k < maxZ + 1; k++) {
	    data[i][j][k] = Double.NaN;
	  }
	}
      }

      // Fill data matrix
      for (int i = 0; i < vals.size(); i++) {
	data[x.get(i)][y.get(i)][z.get(i)] = vals.get(i);
      }

      return data;
    }
    catch (IOException e) {
      throw e;
    }
  }
}
