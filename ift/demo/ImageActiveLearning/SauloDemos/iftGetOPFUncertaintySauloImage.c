#include "ift.h"
#include "iftExperimentUtility.h"
#include "iftSuperpixelFeatures.h"

int main(int argc, char** argv) {
  static   int BG_LABEL = 2;
  static   int OBJ_LABEL = 1;

  if (argc < 5) {
    printf("Usage: %s <orig img> <seg img> <gt img> <opf_classifier.zip>\n", argv[0]);
    exit(-1);
  }

  iftImage *img = iftReadImageByExt(argv[1]);
  iftImage *labelMap = iftReadImageByExt(argv[2]);
  iftImage *gt = iftReadImageByExt(argv[3]);
  iftCplGraph *graph = iftReadCplGraph(argv[4]);

  // --- Feature computation
  int nBinsPerChannel = 20;

  // Build histogram features from superpixels
  fprintf(stderr, "Computing sp histogram...\n");
  iftMatrix *spHistogram = iftComputeSuperpixelFeaturesByColorHistogram(img, labelMap, nBinsPerChannel);
  for (int row = 0; row < spHistogram->nrows; ++row)
    iftNormalizeFeatures(iftMatrixRowPointer(spHistogram, row), spHistogram->ncols);

  // Use overlap for reference to what coincides with ground truth
  float *spOverlap = iftGetSuperpixelObjectOverlap(labelMap, gt, -1);
  int *spSize = iftGetSuperpixelSizes(labelMap, -1);

  // Convert superpixels to iftDataSet
  fprintf(stderr, "Computing sp dataset...\n");
  iftDataSet *spDataSet = iftFeatureMatrixToDataSet(spHistogram);
  for (int i = 0; i < spDataSet->nsamples; ++i) {
    iftSample *s = &(spDataSet->sample[i]); // Shorthand
    s->truelabel = (spOverlap[i] > 0.05) ? OBJ_LABEL : BG_LABEL; 
    s->group = i+1; // sp is identified by positive values
  }
  iftSetStatus(spDataSet, IFT_TEST);
  iftSetDistanceFunction(spDataSet, 10); // Chi-Square

  //printf("Comp dist table...\n");
  //iftDist = iftCompDistanceTable(graph->Z, Ztest);
  fprintf(stderr, "Classification...\n");
  iftClassifyWithCertaintyValues(graph, spDataSet);
  //iftDestroyDistanceTable(&iftDist);
  
  // Check uncertainty of false negatives
  int fpCountByCutoff = 0;
  int fpNoCutoff = 0;
  float cutoffVal = 0.80f;
  for (int i = 0; i < spDataSet->nsamples; ++i) {
    iftSample *s = &(spDataSet->sample[i]); // Shorthand
    // Obtain actual value
    if (s->truelabel == OBJ_LABEL && s->label != s->truelabel && spOverlap[i] > 0.5f) {
      printf("%f\n", s->weight);
    }
    // Manual inspection
    /*
    if (s->truelabel == OBJ_LABEL) {
      if (s->label == s->truelabel)
        printf("True positive sample %d ", i);
      else
        printf("False negative sample %d ", i);
      printf("with overlap %f and size %d has uncertainty %f\n", spOverlap[i], spSize[i], s->weight);

    } */
    if (s->truelabel == BG_LABEL && s->label == s->truelabel) {
      if (s->weight < cutoffVal)
        fpCountByCutoff++;
    }
    if (s->truelabel == BG_LABEL && s->label == OBJ_LABEL)
      fpNoCutoff++;
  }
  fprintf(stderr, "Cutoff %f would introduce %d false positives\n", cutoffVal, fpCountByCutoff);
  //printf("%d,%d\n", fpCountByCutoff, fpNoCutoff);

  iftDestroyMatrix(&spHistogram);
  free(spOverlap);
  free(spSize);
  iftDestroyDataSet(&spDataSet);
  iftDestroyCplGraph(&graph);
  iftDestroyImage(&img);
  iftDestroyImage(&labelMap);
  iftDestroyImage(&gt);

  return 0;
}

