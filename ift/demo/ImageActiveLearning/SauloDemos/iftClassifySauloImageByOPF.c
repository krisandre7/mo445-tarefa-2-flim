#include "ift.h"
#include "iftExperimentUtility.h"
#include "iftSuperpixelFeatures.h"

int main(int argc, char** argv) {
  static   int BG_LABEL = 2;
  static   int OBJ_LABEL = 1;

  if (argc < 6) {
    printf("Usage: %s <orig img> <seg img> <gt img> <opf_classifier.zip> <output>\n", argv[0]);
    exit(-1);
  }

  iftImage *img = iftReadImageByExt(argv[1]);
  iftImage *labelMap = iftReadImageByExt(argv[2]);
  iftImage *gt = iftReadImageByExt(argv[3]);
  iftCplGraph *graph = iftReadCplGraph(argv[4]);

  // --- Feature computation
  int nBinsPerChannel = 20;

  // Build histogram features from superpixels
  printf("Computing sp histogram...\n");
  iftMatrix *spHistogram = iftComputeSuperpixelFeaturesByColorHistogram(img, labelMap, nBinsPerChannel);
  for (int row = 0; row < spHistogram->nrows; ++row)
    iftNormalizeFeatures(iftMatrixRowPointer(spHistogram, row), spHistogram->ncols);

  // Use overlap for reference to what coincides with ground truth
  float *spOverlap = iftGetSuperpixelObjectOverlap(labelMap, gt, -1);

  // Convert superpixels to iftDataSet
  printf("Computing sp dataset...\n");
  iftDataSet *spDataSet = iftFeatureMatrixToDataSet(spHistogram);
  for (int i = 0; i < spDataSet->nsamples; ++i) {
    iftSample *s = &(spDataSet->sample[i]); // Shorthand
    s->truelabel = (spOverlap[i] > IFT_EPSILON) ? OBJ_LABEL : BG_LABEL; 
    s->group = i+1; // sp is identified by positive values
  }
  iftSetStatus(spDataSet, IFT_TEST);
  iftSetDistanceFunction(spDataSet, 10); // Chi-Square

  //printf("Comp dist table...\n");
  //iftDist = iftCompDistanceTable(graph->Z, Ztest);
  printf("Classification...\n");
  iftClassifyWithCertaintyValues(graph, spDataSet);
  //iftDestroyDistanceTable(&iftDist);

  // -- Visualization
  printf("Visualization...\n");
  iftImage *vis = iftCreateImage(img->xsize, img->ysize, img->zsize);
  for (int p = 0; p < vis->n; ++p) {
    int sp = labelMap->val[p] - 1;
    if (spDataSet->sample[sp].label == OBJ_LABEL || spDataSet->sample[sp].weight < 0.75f) {
      vis->val[p] = sp+1; 
    } else {
      vis->val[p] = 0; 
    }
  }
  iftWriteImageByExt(vis, argv[5]);

  iftDestroyMatrix(&spHistogram);
  free(spOverlap);
  iftDestroyDataSet(&spDataSet);
  iftDestroyCplGraph(&graph);
  iftDestroyImage(&vis);
  iftDestroyImage(&img);
  iftDestroyImage(&labelMap);
  iftDestroyImage(&gt);

  return 0;
}

