#include "ift.h"
#include "iftExperimentUtility.h"
#include "iftSuperpixelFeatures.h"

int main(int argc, char** argv) {
  static   int BG_LABEL = 2;
  static   int OBJ_LABEL = 1;

  if (argc < 6) {
    printf("Usage: %s <sample csv> <orig folder> <seg folder> <gt folder> <res dataset.zip>\n", argv[0]);
    exit(-1);
  }

  iftFileSet *fs = iftLoadFileSetFromDirOrCSV(argv[1], 1, true);
  iftDataSet *res = NULL;

  for (int f = 0; f < fs->n; ++f) {
    char *csvPath = fs->files[f]->path;
    char *tmp = iftSplitStringAt(csvPath, IFT_SEP_C, -1);
    char *basename = iftSplitStringAt(tmp, ".", 0);
    free(tmp);
    char *origPath = iftConcatStrings(4, argv[2], IFT_SEP_C, basename, ".png");
    char *segPath = iftConcatStrings(4, argv[3], IFT_SEP_C, basename, ".pgm");
    char *gtPath = iftConcatStrings(4, argv[4], IFT_SEP_C, basename, ".pgm");

    printf("Processing file %s\n", csvPath);

    iftImage *img = iftReadImageByExt(origPath);
    iftImage *labelMap = iftReadImageByExt(segPath);
    iftImage *gt = iftReadImageByExt(gtPath);

    // Label gt connected components
    iftImage *tmpImg = iftForceLabelMapConnectivity(gt, 0);
    iftDestroyImage(&gt);
    gt = tmpImg;

    int nBinsPerChannel = 20;

    // Build histogram features from ground truth regions
    printf("Computing gt histogram...\n");
    iftMatrix *gtHistogram = iftComputeSuperpixelFeaturesByColorHistogram(img, gt, nBinsPerChannel);
    for (int row = 0; row < gtHistogram->nrows; ++row)
      iftNormalizeFeatures(iftMatrixRowPointer(gtHistogram, row), gtHistogram->ncols);
    iftMatrix *gtHistNoBg = iftCreateMatrixPointer(iftMatrixRowPointer(gtHistogram, 1), gtHistogram->ncols, gtHistogram->nrows - 1);

    // Convert gt regions to iftDataSet
    printf("Computing gt dataset...\n");
    iftDataSet *gtDataSet = iftFeatureMatrixToDataSet(gtHistNoBg);
    for (int i = 0; i < gtDataSet->nsamples; ++i) {
      iftSample *s = &(gtDataSet->sample[i]);
      s->truelabel = OBJ_LABEL;
      s->id = f;
      s->group = -(i + 1); // gt is identified by negative value
    }

    // Build histogram features from superpixels
    printf("Computing sp histogram...\n");
    iftMatrix *spHistogram = iftComputeSuperpixelFeaturesByColorHistogram(img, labelMap, nBinsPerChannel);
    for (int row = 0; row < spHistogram->nrows; ++row)
      iftNormalizeFeatures(iftMatrixRowPointer(spHistogram, row), spHistogram->ncols);

    // Use overlap for reference to what coincides with ground truth
    float *spOverlap = iftGetSuperpixelObjectOverlap(labelMap, gt, -1);

    static   float MIN_POSITIVE_OVERLAP = 0.75;
    static   float MAX_NEGATIVE_OVERLAP = 0.00;

    // Count number of useful superpixels
    int unambiguousCount = 0;
    for (int row = 0; row < spHistogram->nrows; ++row)
      if (spOverlap[row] <= MAX_NEGATIVE_OVERLAP || spOverlap[row] >= MIN_POSITIVE_OVERLAP)
        unambiguousCount += 1;

    if (unambiguousCount != spHistogram->nrows)
      printf("Discarding %d ambiguous superpixels out of %d.\n", 
          spHistogram->nrows - unambiguousCount, spHistogram->nrows);

    // Build trimmed matrix with labels
    iftMatrix *spHistogramSelection = iftCreateMatrix(spHistogram->ncols, unambiguousCount);
    int *spLabel = iftAllocIntArray(unambiguousCount);
    int *spId = iftAllocIntArray(unambiguousCount);
    int selectionRow = 0;
    for (int row = 0; row < spHistogram->nrows; ++row) {
      if (spOverlap[row] <= MAX_NEGATIVE_OVERLAP || spOverlap[row] >= MIN_POSITIVE_OVERLAP) {
        spLabel[selectionRow] = (spOverlap[row] >= MIN_POSITIVE_OVERLAP) ? OBJ_LABEL : BG_LABEL;
        spId[selectionRow] = row + 1;
        for (int col = 0; col < spHistogram->ncols; ++col)
          iftMatrixElem(spHistogramSelection, col, selectionRow) = 
            iftMatrixElem(spHistogram, col, row); 
        selectionRow += 1;
      }
    }
    int nSpObj = 0;
    for (int i = 0; i < spHistogram->nrows; ++i)
      if (spLabel[i] == OBJ_LABEL)
        nSpObj++;
    printf("Number of superpixels labeled as obj = %d\n", nSpObj);

    // Convert superpixels to iftDataSet
    printf("Computing sp dataset...\n");
    iftDataSet *spDataSet = iftFeatureMatrixToDataSet(spHistogramSelection);
    for (int i = 0; i < spDataSet->nsamples; ++i) {
      iftSample *s = &(spDataSet->sample[i]); // Shorthand
      s->truelabel = spLabel[i]; 
      s->id = f;
      s->group = spId[i]; // sp is identified by positive values
    }

    iftDataSet *Z = iftMergeDataSets(gtDataSet, spDataSet);
    if (res == NULL) {
      res = Z;
    } else {
      iftDataSet *Z2 = res;
      res = iftMergeDataSets(Z2,Z);
      iftDestroyDataSet(&Z);
      iftDestroyDataSet(&Z2);
    }

    free(csvPath);
    free(basename);
    free(origPath);
    free(segPath);
    free(gtPath);
    iftDestroyMatrix(&gtHistogram);
    iftDestroyDataSet(&gtDataSet);
    iftDestroyMatrix(&spHistogram);
    free(spOverlap);
    iftDestroyMatrix(&spHistogramSelection);
    free(spLabel);
    free(spId);
    iftDestroyDataSet(&spDataSet);
  }

  iftSetDistanceFunction(res, 10); // Chi-Square
  iftCopyRefData(res, (void *) fs, IFT_REF_DATA_FILESET);
  iftWriteDataSet(res, argv[5]);

  iftDestroyDataSet(&res);

  return 0;
}

