#include "ift.h"
#include "iftRandomSelector.h"
#include "iftSuperpixelFeatures.h"
#include "iftExperimentUtility.h"

void iftSelectSamplesByUncertainty(iftDataSet *Z,   iftImage *spLabelMap,   iftImage *prevSegMap, int nSelect, int *selectFlag);
void iftSelectSamplesByRandomSp(iftDataSet *Z,   iftImage *spLabelMap,   iftImage *prevSegMap, int nSelect, int *selectFlag);

int main(int argc, char* argv[])
{
  if (argc < 5) {
    fprintf(stderr, "Usage: %s <baseImg> <segLabelMap> <classifiedPixels.zip> <nActiveIters> [prevSegLabelMap]\n", argv[0]);
    return -1;
  }

  iftImage *img = iftReadImageByExt(argv[1]);
  iftImage *superpixelLabelMap = iftReadImageByExt(argv[2]);
  iftDataSet *Z = iftReadDataSet(argv[3]);
  int nActiveIters = atol(argv[4]);
  iftImage *prevSegLabelMap = argc < 6 ? NULL : iftReadImageByExt(argv[5]);
  int *selectFlag = iftAllocIntArray(iftMaximumValue(superpixelLabelMap));

  for (int t = 0; t <= nActiveIters; ++t) {
    // Train new classifier with currently supervised samples
    printf("Starting active learning iteration %d:\n", t);
    iftCplGraph *graph = iftCreateCplGraph(Z);
    printf("Training with nTrainSamples = %d:\n", Z->ntrainsamples);
    iftSupTrain(graph);
    for (int s = 0; s < Z->nsamples; ++s)
      Z->sample[s].status &= ~(ERROR);
    printf("Classifying:\n");
    iftClassify(graph, Z);

    // Evaluate current classifier
    printf("Evaluating classification:\n");
    float kappa = iftCohenKappaScore(Z);
    printf("Iter %d kappa: %f\n", t, kappa);

    printf("Selecting new samples:\n");
    int nWindows = 1;
    iftSelectSamplesByUncertainty(Z, superpixelLabelMap, 
    //iftSelectSamplesByRandomSp(Z, superpixelLabelMap, 
        prevSegLabelMap, nWindows, selectFlag);

    printf("Clean up\n");
    // Clean up
    iftDestroyCplGraph(&graph);
  }

  // Clean up
  iftDestroyImage(&img);
  iftDestroyImage(&superpixelLabelMap);
  iftDestroyDataSet(&Z);
  iftDestroyImage(&prevSegLabelMap);

  return 0;
}

void iftSelectSamplesByUncertainty(iftDataSet *Z,   iftImage *spLabelMap,   iftImage *prevSegMap, int nSelect, int *selectFlag)
{
  int nSuperpixels = iftMaximumValue(spLabelMap);
  iftMatrix *mx = iftSuperpixelClassificationResults(Z, spLabelMap, prevSegMap);

  // Sort superpixels based on classification results
  int *cIdx = iftAllocIntArray(nSuperpixels);
  float *cErrorMetric = iftAllocFloatArray(nSuperpixels);
  for (int i = 0; i < nSuperpixels; ++i) {
    cIdx[i] = i;
    // How "random" the classification result of the superpixel was
    cErrorMetric[i] = 1.0 - iftMatrixElem(mx, 0, i+1);
  }
  iftFHeapSort(cErrorMetric, cIdx, nSuperpixels, IFT_DECREASING);

  // Perform random selection weighted by order
  while (nSelect > 0 && Z->ntrainsamples < Z->nsamples) {
    int s = iftRandomSelectionWeightedByOrder(nSuperpixels);
    if (selectFlag[cIdx[s]])
      continue;

    selectFlag[cIdx[s]] = 1;
    nSelect -= 1;
    for (int p = 0; p < spLabelMap->n; ++p) {
      if (spLabelMap->val[p] != cIdx[s])
        continue;

      int t = (prevSegMap == NULL) ? p : prevSegMap->val[p] - 1;
      if (!(Z->sample[t].status & TRAIN)) {
        Z->sample[t].label = Z->sample[t].truelabel;
        Z->sample[t].status = TRAIN | SUPERVISED;
        Z->ntrainsamples += 1;
      }
    }
  }

  // Clean up
  iftDestroyMatrix(&mx);
  free(cIdx);
  free(cErrorMetric);
}

void iftSelectSamplesByRandomSp(iftDataSet *Z,   iftImage *spLabelMap,   iftImage *prevSegMap, int nSelect, int *selectFlag)
{
  int nSuperpixels = iftMaximumValue(spLabelMap);
  iftRandomSelector *rs = iftCreateRandomSelectorDefaultValues(nSuperpixels);

  while (nSelect > 0 && Z->ntrainsamples < Z->nsamples) {
    int s = iftPickFromRandomSelector(rs, false);
    if (selectFlag[s])
      continue;

    selectFlag[s] = 1;
    nSelect -= 1;
    for (int p = 0; p < spLabelMap->n; ++p) {
      if (spLabelMap->val[p] != s)
        continue;

      int t = (prevSegMap == NULL) ? p : prevSegMap->val[p] - 1;
      if (!(Z->sample[t].status & TRAIN)) {
        Z->sample[t].label = Z->sample[t].truelabel;
        Z->sample[t].status = TRAIN | SUPERVISED;
        Z->ntrainsamples += 1;
      }
    }
  }

  iftDestroyRandomSelector(&rs);
}

