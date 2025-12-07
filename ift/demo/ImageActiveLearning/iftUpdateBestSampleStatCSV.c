/*
 * @file
 * @brief Finds per sample improvements for OIS metrics.
 * @author Felipe Galvao
 * @data October, 2019
 */
#include "ift.h"
#include "iftExperimentUtility.h"

int findCSVCol(iftCSV *csv, const char *name)
{
  for (int col = 0; col < csv->ncols; ++col) {
    if (strcmp(name, csv->data[0][col]) == 0)
      return col;
  }

  return -1;
}

void copyCSVRow(iftCSV *tgt, iftCSV *src, int row)
{
  for (int col = 0; col < tgt->ncols; ++col)
    strcpy(tgt->data[row][col], src->data[row][col]);
}

int main(int argc, char *argv[])
{
  if (argc < 5) {
    fprintf(stderr, "Usage %s <tgt.csv> <input.csv> <metric_column> <ordering(negative = MIN, MAX otherwise)>\n", argv[0]);
    return -1;
  }

  bool hasHeader = false;
  iftCSV *tgtCsv = iftReadCSVWithHeader(argv[1], ',', &hasHeader);
  if (!hasHeader) {
    fprintf(stderr, "Target csv is missing header!\n");
    return -1;
  }

  iftCSV *inCsv = iftReadCSVWithHeader(argv[2], ',', &hasHeader);
  if (!hasHeader) {
    fprintf(stderr, "Input csv is missing header!\n");
    return -1;
  }

  if (tgtCsv->ncols != inCsv->ncols || tgtCsv->nrows != inCsv->nrows) {
    fprintf(stderr, "Target %ldx%ld csv does not match input %ldx%ld csv!\n",
        tgtCsv->nrows, tgtCsv->ncols, inCsv->nrows, inCsv->ncols);
    return -1;
  }

  for (int col = 0; col < tgtCsv->ncols; ++col) {
    if (strcmp(tgtCsv->data[0][col], inCsv->data[0][col]) != 0) {
      fprintf(stderr, "CSV headers do not match at col %d: \"%s\" vs \"%s\"\n", col, tgtCsv->data[0][col], inCsv->data[0][col]);
      return -1;
    }
  }

  int refCol = findCSVCol(tgtCsv, argv[3]);
  if (refCol < 0) {
    fprintf(stderr, "Can not find column with header \"%s\"!\n", argv[3]);
    return -1;
  }

  bool findMax = atol(argv[4]) < 0 ? false : true;

  for (int row = 1; row < tgtCsv->nrows; ++row) {
    float tgtVal = atof(tgtCsv->data[row][refCol]);
    float inVal = atof(inCsv->data[row][refCol]);

    if (findMax && (inVal > tgtVal))
      copyCSVRow(tgtCsv, inCsv, row);
    else if (!findMax && (inVal < tgtVal))
      copyCSVRow(tgtCsv, inCsv, row);
  }

  iftWriteCSV(tgtCsv, argv[1], ',');

  iftDestroyCSV(&tgtCsv);
  iftDestroyCSV(&inCsv);

  return 0;
}
