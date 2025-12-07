#include "ift.h"

int main(int argc, char *argv[])
{
  // arg1 = base csv
  // arg2 = base csv time col
  // arg3... = time csv in order of level
  
  if (argc < 6) {
    printf("Usage: %s <base.csv> <base time col> <target.csv> <l0.csv> <l0 time col> [<l1.csv> <l1 time col>, ...]", argv[0]);
    return -1;
  }

  iftCSV *base_csv = iftReadCSV(argv[1], ',');
  int baseTimeCol = atol(argv[2]);
  int argOffset = 4; 
  int colOffset = base_csv->ncols;
  int nExtraCols = (argc - argOffset) / 2 + 1; /* Each new time + sum */
  iftCSV *new_csv = iftCreateCSV(base_csv->nrows, base_csv->ncols + nExtraCols);
  float *totalTime = iftAllocFloatArray(base_csv->nrows - 1);

  /* Copy identical section. */
  for (int col = 0; col < base_csv->ncols; ++col) {
    for (int row = 0; row < base_csv->nrows; ++row) {
      strcpy(new_csv->data[row][col], base_csv->data[row][col]);
      if (col == baseTimeCol && row > 0) {
        totalTime[row-1] += atof(base_csv->data[row][col]);
      }
    }
  }

  /* Fill previous levels times. */
  for (int col = 0; col < nExtraCols - 1; ++col) {
    iftCSV *time_csv = iftReadCSV(argv[argOffset + col*2], ',');
    int levelTimeCol = atol(argv[argOffset + col*2 + 1]);
    char strBuffer[IFT_STR_DEFAULT_SIZE];
    sprintf(strBuffer, "Level %d time", col);
    strcpy(new_csv->data[0][col + colOffset], strBuffer);
    for (int row = 1; row < base_csv->nrows; ++row) {
      strcpy(new_csv->data[row][col + colOffset], time_csv->data[1][levelTimeCol]);
      totalTime[row-1] += atof(time_csv->data[1][levelTimeCol]);
    }
    iftDestroyCSV(&time_csv);
  }

  /* Fill last col with total time. */
  int finalCol = nExtraCols + colOffset - 1;
  strcpy(new_csv->data[0][finalCol], "Total time");
  for (int row = 1; row < base_csv->nrows; ++row) {
    sprintf(new_csv->data[row][finalCol], "%f", totalTime[row-1]);
  }

  iftWriteCSV(new_csv, argv[3], ',');

  iftDestroyCSV(&base_csv);
  iftDestroyCSV(&new_csv);
}
