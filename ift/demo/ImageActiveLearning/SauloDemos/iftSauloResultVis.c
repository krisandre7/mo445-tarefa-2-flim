#include <ift.h>
#include "iftExperimentUtility.h"

iftColor GetYellow();
iftColor GetRed();
iftColor GetGreen();

#define COLOR_DEPTH (16)
#define COLOR_MAXVAL ((1 << COLOR_DEPTH) - 1)

int main(int argc, char *argv[])
{
  if (argc < 6) {
    printf("Usage: %s <fileset.csv> <classification_results.csv>", argv[0]);
    printf(" <patches_info.csv> <orig folder> <gt folder> <result folder>\n");
    return -1;
  }
  
  // Overlay original images with groundtruth into result folder
  iftCSV *csvOrig = iftReadCSV(argv[1], ',');
  for (int row = 0; row < csvOrig->nrows; ++row) {
    char *path = csvOrig->data[row][0];
    char *basename = iftFilename(path, iftFileExt(path));
    iftImage *orig = iftReadImageByExt("%s/%s.png", argv[4], basename);
    iftImage *gt = iftReadImageByExt("%s/%s.pgm", argv[5], basename);

    printf("Overlaying groundtruth on %s (%d/%ld)\n",
        basename, row+1, csvOrig->nrows);

    iftColor color = GetYellow();
    iftImage *res = iftOverlaySegmentationBorders(orig, gt, color);
    //iftImage *res = iftCopyImage(orig);
    iftWriteImageByExt(res, "%s/%06d.png", argv[6], csvOrig->nrows - row); 

    free(basename);
    iftDestroyImage(&orig);
    iftDestroyImage(&gt);
    iftDestroyImage(&orig);
  }
  iftDestroyCSV(&csvOrig);

  // Overlay patches classified as positive
  iftCSV *csvClassification = iftReadCSV(argv[2], ',');
  iftCSV *csvPatchData = iftReadCSV(argv[3], ',');
  iftAdjRel *boxBrush = iftCircular(1.0f);
  assert(csvClassification->ncols == 4);
  assert(csvPatchData->ncols == 5);
  for (int row = 0; row < csvClassification->nrows; ++row) {
    float probPositive = atof(csvClassification->data[row][1]);
    float probNegative = atof(csvClassification->data[row][2]);

    if (probNegative >= probPositive)
      continue;

    printf("Positive patch %s!\n", csvClassification->data[row][0]);

    int patchClass = atoi(csvClassification->data[row][3]);
    iftColor boxColor = (patchClass == 1 ? GetGreen() : GetRed());

    printf("Reading tgt image\n");
    char *path = csvClassification->data[row][0];
    char *basename = iftFilename(path, iftFileExt(path));
    char *str = iftSplitStringAt(basename, "_", 2);
    int tgtImg = atoi(str);
    printf("Path: %s/%06d.png\n", argv[6], tgtImg);
    iftImage *res = iftReadImageByExt("%s/%06d.png", argv[6], tgtImg);
    
    printf("Building box\n");
    int x0 = atoi(csvPatchData->data[row][3]);
    int y0 = atoi(csvPatchData->data[row][4]);
    iftVoxel begin = { x0, y0, 0};
    iftVoxel end = { x0 + 99, y0 + 99, 0};
    iftBoundingBox bb = { begin, end };

    printf("Draw box\n");
    iftDrawBoundingBoxBordersInPlace(res, bb, boxColor, boxBrush, false);
    iftWriteImageByExt(res, "%s/%06d.png", argv[6], tgtImg);

    printf("Clean\n");
    free(basename);
    free(str);
    iftDestroyImage(&res);
  }
  iftDestroyCSV(&csvClassification);
  iftDestroyCSV(&csvPatchData);
  iftDestroyAdjRel(&boxBrush);

  return 0;
}

iftColor GetYellow()
{
  iftColor color;
  color.val[0] = COLOR_MAXVAL;
  color.val[1] = COLOR_MAXVAL;
  color.val[2] = 0;
  color.alpha = 1.0f;

  return iftRGBtoYCbCr(color, COLOR_MAXVAL);
}

iftColor GetRed()
{
  iftColor color;
  color.val[0] = COLOR_MAXVAL;
  color.val[1] = 0;
  color.val[2] = 0;
  color.alpha = 1.0f;

  return iftRGBtoYCbCr(color, COLOR_MAXVAL);
}

iftColor GetGreen()
{
  iftColor color;
  color.val[0] = 0;
  color.val[1] = COLOR_MAXVAL;
  color.val[2] = 0;
  color.alpha = 1.0f;

  return iftRGBtoYCbCr(color, COLOR_MAXVAL);
}
