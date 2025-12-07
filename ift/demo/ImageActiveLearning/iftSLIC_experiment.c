/**
 * @file
 * @brief Full test against multiple parameters.
 * @author Felipe Lemes Galvao
 * @date August 1, 2017
 *
 */

#include <ift.h>
#include "iftBoVW.h"
#include "iftSuperpixelFeatures.h"
#include "iftExperimentUtility.h"

int main(int argc, char* argv[])
{
  if (argc < 5) 
    iftError("Usage: iftSLIC_experiment <path/to/images> <path/to/groundtruth> <path/to/results> <nSuperpixels>", "main");

  // Check if supplied paths exist
  for (int path = 1; path < 3; ++path) {
    if (!iftDirExists(argv[path]))
      iftError("Path %s does not exist.", "main", argv[path]);
  }
  iftMakeDir(argv[3]);

  iftDir *origDir = iftLoadDir(argv[1], 1);

#define PARAM1_SIZE 1
  float alphaArr[PARAM1_SIZE];
  for (int i = 0; i < PARAM1_SIZE; ++i)
    alphaArr[i] = 80.0 + i * 2.0;

  iftCSV *superMx = iftCreateCSV(13, PARAM1_SIZE + 1);
  sprintf(superMx->data[0][0], "brMean");
  sprintf(superMx->data[1][0], "brStdDev");
  sprintf(superMx->data[2][0], "ueMean");
  sprintf(superMx->data[3][0], "ueStdDev");
  sprintf(superMx->data[4][0], "compMean");
  sprintf(superMx->data[5][0], "compStdDev");
  sprintf(superMx->data[6][0], "topMean");
  sprintf(superMx->data[7][0], "topStdDev");
  sprintf(superMx->data[8][0], "nSpMean");
  sprintf(superMx->data[9][0], "nSpStdDev");
  sprintf(superMx->data[10][0], "diceMean");
  sprintf(superMx->data[11][0], "diceStdDev");
  sprintf(superMx->data[12][0], "slic m");

  for (int p1 = 0; p1 < PARAM1_SIZE; ++p1) {
    printf("Start m = %f:\n", alphaArr[p1]);
    int superIdx = p1 + 1;
    sprintf(superMx->data[12][superIdx], "%f", alphaArr[p1]);

    // Metrics
    int nSamples = origDir->nfiles;
    float *br = iftAllocFloatArray(nSamples);
    float *ue = iftAllocFloatArray(nSamples);
    float *comp = iftAllocFloatArray(nSamples);
    float *top = iftAllocFloatArray(nSamples);
    float *nSp = iftAllocFloatArray(nSamples);
    float *dice = iftAllocFloatArray(nSamples);

    for (int s = 0; s < nSamples; ++s) {
      printf("Sample %d/%d\n", s+1, nSamples);
      // Load Image
      char *imgPath = origDir->files[s]->path;
      char *imgFilename = iftFilename(imgPath, NULL);
      imgFilename = strtok(imgFilename, ".");
      iftImage *img = iftReadImageByExt(imgPath);
      bool is3D = iftIs3DImage(img);

      // Compute ISF
      int nSuperpixels = atoi(argv[4]);
      int nIters = 10;
      float mSlic = alphaArr[p1];
      int imgRange = iftNormalizationValue(iftMaximumValue(img));
      iftImage *superpixelLabelMap = iftGenerateSuperpixelsBySlic(img, NULL,  nSuperpixels, mSlic, imgRange, NULL);
      iftImage *superpixelBorders = iftBorderImage(superpixelLabelMap, 0);

      // Write segmentation result to file
      char resFilename[256];
      sprintf(resFilename, "%s_%d.%s", imgFilename, p1, is3D ? "scn" : "pgm"); 
      char *resPath = iftJoinPathnames(2, argv[3], resFilename);
      iftWriteImageByExt(superpixelLabelMap, resPath);
      if (!is3D) {
        // Create a nice visualization to check results on 2D images
        iftImage *tmpImg = iftOverlaySegmentationBorders(img, superpixelBorders);
        sprintf(resFilename, "%s_%d_visualization.%s", imgFilename, p1, "png");
        free(resPath);
        resPath = iftJoinPathnames(2, argv[3], resFilename);
        iftWriteImageByExt(tmpImg, resPath);
        iftDestroyImage(&tmpImg);
      }

      // Check results against one or more groundtruths
      char gtRegex[256];
      sprintf(gtRegex, "%s.*\\.%s", imgFilename, is3D ? "scn" : "pgm");
      iftDir *gtDir = iftLoadFilesFromDirByRegex(argv[2], gtRegex); 
      assert(gtDir->nfiles > 0);
      br[s] = ue[s] = dice[s] = 0.0;
      for (int g = 0; g < gtDir->nfiles; ++g) {
        iftImage *gtLabelMap = iftReadImageByExt(gtDir->files[g]->path);
        iftImage *gtBorders = iftBorderImage(gtLabelMap, 0);
        iftImage *bestSegmentation = iftSuperpixelToMajoritySegmentation(superpixelLabelMap, gtLabelMap);

        br[s] += iftBoundaryRecall(gtBorders, superpixelBorders, 1.0);
        ue[s] += iftUnderSegmentation(gtLabelMap, superpixelLabelMap);
        float *diceArr = iftFScoreMultiLabel(bestSegmentation, gtLabelMap, iftMaximumValue(gtLabelMap));
        dice[s] += diceArr[0]; 
        iftDestroyImage(&gtLabelMap);
        iftDestroyImage(&gtBorders);
        iftDestroyImage(&bestSegmentation);
        free(diceArr);
      }
      br[s] /= gtDir->nfiles;
      ue[s] /= gtDir->nfiles;
      dice[s] /= gtDir->nfiles;
      printf("BR = %f\n", br[s]);
      printf("Dice = %f\n", dice[s]);

      // Superpixel properties
      nSp[s] = (float) iftMaximumValue(superpixelLabelMap);
      printf("nSp = %f\n", nSp[s]);
      comp[s] = is3D ? 0.0 : iftCompactness2D(superpixelLabelMap);
      top[s] = iftTopologyMeasure(superpixelLabelMap);

      // Clean up
      free(imgFilename);
      iftDestroyImage(&img);
      iftDestroyImage(&superpixelLabelMap);
      iftDestroyImage(&superpixelBorders);
      free(resPath);
      iftDestroyDir(&gtDir);
    }
    printf("\n");

    // Mean + std deviation for all metrics
    float brMean = iftMeanFloatArray(br, nSamples);
    float brStdDev = iftStddevFloatArray(br, nSamples);
    float ueMean = iftMeanFloatArray(ue, nSamples);
    float ueStdDev = iftStddevFloatArray(ue, nSamples);
    float compMean = iftMeanFloatArray(comp, nSamples);
    float compStdDev = iftStddevFloatArray(comp, nSamples);
    float topMean = iftMeanFloatArray(top, nSamples);
    float topStdDev = iftStddevFloatArray(top, nSamples);
    float nSpMean = iftMeanFloatArray(nSp, nSamples);
    float nSpStdDev = iftStddevFloatArray(nSp, nSamples);
    float diceMean = iftMeanFloatArray(dice, nSamples);
    float diceStdDev = iftStddevFloatArray(dice, nSamples);

    printf("Results:\n");
    printf("BR: %f +- %f \n", brMean, brStdDev);
    sprintf(superMx->data[0][superIdx], "%f", brMean);
    sprintf(superMx->data[1][superIdx], "%f", brStdDev);
    printf("UE: %f +- %f \n", ueMean, ueStdDev);
    sprintf(superMx->data[2][superIdx], "%f", ueMean);
    sprintf(superMx->data[3][superIdx], "%f", ueStdDev);
    printf("Comp: %f +- %f \n", compMean, compStdDev);
    sprintf(superMx->data[4][superIdx], "%f", compMean);
    sprintf(superMx->data[5][superIdx], "%f", compStdDev);
    printf("Top: %f +- %f \n", topMean, topStdDev);
    sprintf(superMx->data[6][superIdx], "%f", topMean);
    sprintf(superMx->data[7][superIdx], "%f", topStdDev);
    printf("nSp: %f +- %f \n", nSpMean, nSpStdDev); 
    sprintf(superMx->data[8][superIdx], "%f", nSpMean);
    sprintf(superMx->data[9][superIdx], "%f", nSpStdDev);
    printf("Dice: %f +- %f \n", diceMean, diceStdDev);
    sprintf(superMx->data[10][superIdx], "%f", diceMean);
    sprintf(superMx->data[11][superIdx], "%f", diceStdDev);

    free(br);
    free(ue);
    free(comp);
    free(top);
    free(nSp);
    free(dice);
  }

  char *csvSuperPath = iftJoinPathnames(2, argv[3], "results.csv");
  iftWriteCSV(superMx, csvSuperPath, ',');

  free(csvSuperPath);
  iftDestroyCSV(&superMx);
  iftDestroyDir(&origDir);
  return 0;
}
