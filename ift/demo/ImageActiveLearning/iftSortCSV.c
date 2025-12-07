#include <ift.h>
#include "iftExperimentUtility.h"

int main(int argc, char *argv[])
{
  if (argc < 4) {
    printf("Usage: iftSortCSV <input.csv> <index col> <output.csv>\n");
    return -1;
  }

  bool hasHeader = false;
  iftCSV *csv = iftReadCSVWithHeader(argv[1], ',', &hasHeader);
  int sortCol = atol(argv[2]);
  int rowIgnore = hasHeader ? 1 : 0;
  int size = csv->nrows - rowIgnore;

  // Read CSV data
  float *refData = iftAllocFloatArray(size);
  int *refIdx = iftAllocIntArray(size); 
  for (int row = rowIgnore; row < csv->nrows; ++row) {
    refData[row - rowIgnore] = atof(csv->data[row][sortCol]);
    refIdx[row - rowIgnore] = row;
  }

  iftFHeapSort(refData, refIdx, size, IFT_INCREASING);

  // Build new csv based on sorted values
  iftCSV *sortedCsv = iftCreateCSV(csv->nrows, csv->ncols);
  for (int row = 0; row < rowIgnore; ++row)
    for (int col = 0; col < csv->ncols; ++col)
      strcpy(sortedCsv->data[row][col], csv->data[row][col]);
  for (int row = rowIgnore; row < csv->nrows; ++row)
    for (int col = 0; col < csv->ncols; ++col)
      strcpy(sortedCsv->data[row][col], csv->data[refIdx[row-rowIgnore]][col]);

  iftWriteCSV(sortedCsv, argv[3], ',');

  // Clean up
  iftDestroyCSV(&csv);
  iftDestroyCSV(&sortedCsv);
  free(refData);
  free(refIdx);
  
  return 0;
}
