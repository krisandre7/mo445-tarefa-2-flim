#include <ift.h>

#define FIELD_COUNT 4

int main(int argc, char * argv[])
{
  //$BIN_AGGREGATE $CSV_PREFIX $ITERS $RESULT_PATH
  if (argc < 4) {
    printf("Usage: %s <csv_prefix> <num_of_files> <output_path>\n", argv[0]);
    return 1;
  }

  char pathBuffer[512] = "";
  int numOfFiles = atoi(argv[2]);
  iftCSV ** files = calloc(numOfFiles, sizeof(*files));

  for (int i = 0; i < numOfFiles; ++i) { 
    sprintf(pathBuffer, "%s_%d.csv", argv[1], i);
    files[i] = iftReadCSV(pathBuffer, ',');
  }

  int numIters = files[0]->nrows-1;

  iftCSV * resultsCSV = iftCreateCSV(numIters+1, 9);
  strcpy(resultsCSV->data[0][0], "Iteration");
  strcpy(resultsCSV->data[0][1], "Accuracy");
  strcpy(resultsCSV->data[0][2], "Accuracy std");
  strcpy(resultsCSV->data[0][3], "Annotated samples");
  strcpy(resultsCSV->data[0][4], "Annotated samples std");
  strcpy(resultsCSV->data[0][5], "Execution time");
  strcpy(resultsCSV->data[0][6], "Execution time std");
  strcpy(resultsCSV->data[0][7], "Missing classes");
  strcpy(resultsCSV->data[0][8], "Missing classes std");

  for (int iter = 1; iter <= numIters; ++iter) {
    sprintf(resultsCSV->data[iter][0], "%d", iter);

    double mean[FIELD_COUNT] = {0.0, 0.0, 0.0, 0.0};
    double std[FIELD_COUNT] = {0.0, 0.0, 0.0, 0.0};

    // TODO Deal with empty rows

    for (int i = 0; i < numOfFiles; ++i)
      for (int f = 0; f < FIELD_COUNT; ++f)
        mean[f] += atof(files[i]->data[iter][f+1]);

    for (int f = 0; f < FIELD_COUNT; ++f)
      mean[f] /= numOfFiles;

    for (int i = 0; i < numOfFiles; ++i) {
      for (int f = 0; f < FIELD_COUNT; ++f) {
        double diff = mean[f] - atof(files[i]->data[iter][f+1]);
        std[f] += diff * diff;
      }
    }

    for (int f = 0; f < FIELD_COUNT; ++f) {
      std[f] /= numOfFiles;
      std[f] = sqrt(std[f]);
    }

    for (int f = 0; f < FIELD_COUNT; ++f) {
      sprintf(resultsCSV->data[iter][f*2 + 1], "%.5f", mean[f]);
      sprintf(resultsCSV->data[iter][f*2 + 2], "%.5f", std[f]);
    }
  }

  iftWriteCSV(resultsCSV, argv[3], ',');

  // Clean up
  iftDestroyCSV(&resultsCSV);
  for (int i = 0; i < numOfFiles; ++i)
    iftDestroyCSV(&(files[i]));
  free(files);

  return 0;
}
