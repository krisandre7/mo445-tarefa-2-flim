/*
 * @file
 * @brief Compute first and second order stats of sample-indexed csv.
 * @author Felipe Lemes Galvao
 * @date December, 2017
 */
#include "ift.h"
#include "iftExperimentUtility.h"

bool isNumericField(const char *str)
{
  if (str == NULL)
    return false;

  int i = 0;
  while (str[i] != '\0') {
    if (!isdigit(str[i]) && str[i] != '.')
      return false;
    i++;
  }
  return true;
}

int main(int argc, char *argv[])
{
  if (argc < 3)
    iftError("Usage iftAggregateCSVStats <data.csv> <print-options>", "main");

  bool hasHeader = false;
  iftCSV *csv = iftReadCSVWithHeader(argv[1], ',', &hasHeader);
  int printOpt = atol(argv[2]);

  int firstDataRow = hasHeader ? 1 : 0;
  if (csv->nrows < firstDataRow + 1)
    iftError("CSV has no data.", "main");

  bool isFirstCol = true;
  float *val = iftAllocFloatArray(csv->nrows - firstDataRow);
  for (int col = 0; col < csv->ncols; ++col) {
    // Skip columns without numeric data
    if (!isNumericField(csv->data[firstDataRow][col]))
      continue;

    if (printOpt == 2 && !isFirstCol) {
      printf(",");
    }
    isFirstCol = false;

    for (int row = firstDataRow; row < csv->nrows; ++row)
      val[row - firstDataRow] = atof(csv->data[row][col]);

    float mean = iftMeanFloatArray(val, csv->nrows - firstDataRow);
    float stdDev = iftStddevFloatArray(val, csv->nrows - firstDataRow);

    if (printOpt == 1) {
      if (hasHeader)
        printf("%s = ", csv->data[0][col]);
      else
        printf("col %d = ", col);
      printf("%f+-%f\n", mean, stdDev);
    }

    if (printOpt == 2)
      printf("%f,%f", mean, stdDev);
  }

  free(val);
  iftDestroyCSV(&csv);

  return 0;
}
