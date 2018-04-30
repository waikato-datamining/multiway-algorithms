package nz.ac.waikato.cms.adams.multiway.data;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.ArrayUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Data reader class.
 *
 * @author Steven Lang
 */
public class DataReader {

  /**
   * Read three-way sparse data of the following format: x0 y0 z0 value0 x0 y0 z1 value1 ...
   *
   * <p>Missing points default to {@link Double#NaN}.
   *
   * @param path Path to data
   * @param sep CSV separator
   * @param valueIdx Data value index
   * @param hasHeader File includes a header at the top
   * @return Data array
   * @throws IOException Could not access given path
   */
  public static double[][][] read3WaySparse(
      String path, String sep, int valueIdx, boolean hasHeader) throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(path))) {
      String line;
      int rowCount = 0;
      List<Integer> x = new ArrayList<>();
      List<Integer> y = new ArrayList<>();
      List<Integer> z = new ArrayList<>();
      List<Double> vals = new ArrayList<>();

      int[] idxs = new int[] {0, 1, 2, 3};
      ArrayUtils.removeElement(idxs, valueIdx);

      while ((line = br.readLine()) != null) {

        // Skip header if present
        if (hasHeader && rowCount == 0) {
          rowCount++;
          continue;
        } else if (line.isEmpty()) {
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
    } catch (IOException e) {
      throw e;
    }
  }

  /**
   * Read two-way sparse data of the following format: x0 y0 value0 x0 y0 value1 ...
   *
   * <p>Missing points default to {ew@link Double#NaN}.
   *
   * @param path Path to data
   * @param sep CSV separator
   * @param hasHeader File includes a header at the top
   * @return Data array
   * @throws IOException Could not access given path
   */
  public static double[][] readSparseMatrix(String path, String sep, boolean hasHeader)
      throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(path))) {
      String line;
      int rowCount = 0;
      List<Integer> x = new ArrayList<>();
      List<Integer> y = new ArrayList<>();
      List<Integer> z = new ArrayList<>();
      List<Double> vals = new ArrayList<>();

      int[] idxs = new int[] {0, 1, 2};

      while ((line = br.readLine()) != null) {

        // Skip header if present
        if (hasHeader && rowCount == 0) {
          rowCount++;
          continue;
        } else if (line.isEmpty()) {
          continue;
        }

        final String[] split = line.split(sep);
        rowCount++;

        // Add indices and corresponding value
        x.add(Integer.valueOf(split[idxs[0]]));
        y.add(Integer.valueOf(split[idxs[1]]));
        vals.add(Double.valueOf(split[2]));
      }

      // Get maximum indices
      int maxX = Collections.max(x);
      int maxY = Collections.max(y);

      // Init NaN data matrix
      double[][] data = new double[maxX + 1][maxY + 1];
      for (int i = 0; i < maxX + 1; i++) {
        for (int j = 0; j < maxY + 1; j++) {
          data[i][j] = Double.NaN;
        }
      }

      // Fill data matrix
      for (int i = 0; i < vals.size(); i++) {
        data[x.get(i)][y.get(i)] = vals.get(i);
      }

      return data;
    } catch (IOException e) {
      throw e;
    }
  }

  /**
   * Read a three way tensor from different files that are indexed. E.g.: - data0.csv - data1.csv -
   * ... - dataN.csv
   *
   * @param namePrefix Filename prefix before index
   * @param startIdx Index start
   * @param endIdx Index end (inclusive)
   * @param nameSuffix Filename suffix after index
   * @param sep CSV separator
   * @param hasHeader File includes a header at the top
   * @return Data array
   */
  public static double[][][] read3WayMultiCsv(
      String namePrefix, String nameSuffix, int startIdx, int endIdx, String sep, boolean hasHeader)
      throws IOException {

    int numComponents = endIdx - startIdx + 1;
    int numRows = 0;
    int numColumns = 0;
    double[][][] data = null;

    for (int i = startIdx; i <= endIdx; i++) {
      String path = namePrefix + i + nameSuffix;
      List<String> lines = FileUtils.readLines(new File(path));

      // Remove first line if file has header
      if (hasHeader) {
        lines.remove(0);
      }

      // Init data array
      if (data == null) {
        // Parse lines
        numRows = lines.size();
        // TODO: Check if lines are empty
        numColumns = lines.get(0).split(sep).length;

        data = new double[numRows][numColumns][numComponents];
      }

      double[][] componentI =
          lines
              .stream()
              .map(s -> s.split(sep))
              .map(s -> Arrays.stream(s).mapToDouble(Double::parseDouble).toArray())
              .toArray(double[][]::new);

      final int offsetCorrectedIdx = i - startIdx;
      fillComponent(data, componentI, numRows, numColumns, offsetCorrectedIdx);
    }

    return data;
  }

  /**
   * Simulate data[:][:][idxComponent] = matrix[:][:]
   *
   * @param data Data tensor
   * @param matrix Matrix to assign at data[:][:][idxComponent]
   * @param numRows data.length
   * @param numColumns data[i].length
   * @param idxComponent Component index
   */
  private static void fillComponent(
      double[][][] data, double[][] matrix, int numRows, int numColumns, int idxComponent) {
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numColumns; j++) {
        data[i][j][idxComponent] = matrix[i][j];
      }
    }
  }
}
