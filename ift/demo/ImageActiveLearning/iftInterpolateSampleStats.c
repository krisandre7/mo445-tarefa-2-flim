#include <ift.h>
#include <iftExperimentUtility.h>

int main(int argc, char *argv[])
{
  iftCSV *csv = iftReadCSV(argv[1], ',');
  int targetSp = atol(argv[2]);
  int spCol = 1; // We are assuming a specific csv formatting
  int size = csv->nrows - 1;

  int *spArray = iftAllocIntArray(size);
  for (int row = 1; row < csv->nrows; ++row)
    spArray[row-1] = atol(csv->data[row][spCol]);

  // Sanity test, rows should be already ordered by sp count
  for (int i = 1; i < size; ++i)
    if (spArray[i] < spArray[i-1])
      return -1;
  // Target must be within range
  if (targetSp < spArray[0] || targetSp > spArray[size-1])
    return -1;

  // Find the correct interval bounds
  int lowerBound = 0;
  int upperBound = 0;
  float alpha = 0.0;
  for (int i = 1; i < size; ++i) {
    if (targetSp <= spArray[i]) {
      lowerBound = i - 1;
      upperBound = i;
      alpha = (float)(targetSp - spArray[lowerBound]) /
        (float)(spArray[upperBound] - spArray[lowerBound]);
      break;
    }
  }

  // Print interpolated result
  for (int col = 0; col < csv->ncols; ++col) {
    if (col == spCol) {
      printf("%d", targetSp);
    } else {
      float lowerStat = atof(csv->data[lowerBound+1][col]);
      float upperStat = atof(csv->data[upperBound+1][col]);
      printf("%f", lowerStat + alpha * (upperStat - lowerStat));
    }

    if (col < csv->ncols - 1)
      printf(",");
  }

  return 0;
}
