#include "ift.h"
#include "iftExperimentUtility.h"
#include "iftSuperpixelFeatures.h"

static   int TARGET_RADIUS = 49;
static   int BG_LABEL = 2;
static   int OBJ_LABEL = 1;
static   int MAX_CANDIDATE_SIZE = 3000; // should match iftSizeFilteringMerge

iftBoundingBox iftGetBoundingBoxFromCenter(int x, int y,   iftImage *img)
{
  iftBoundingBox bb = {
    .begin = {
      .x = x - TARGET_RADIUS,
      .y = y - TARGET_RADIUS,
      .z = 0},
    .end   = {
      .x = x + TARGET_RADIUS + 1,
      .y = y + TARGET_RADIUS + 1,
      .z = 0}
  };

  // Force bounding box into image domain
  if (bb.begin.x < 0) {
    int offset = -bb.begin.x;
    bb.begin.x += offset;
    bb.end.x += offset;
  }
  if (bb.begin.y < 0) {
    int offset = -bb.begin.y;
    bb.begin.y += offset;
    bb.end.y += offset;
  }
  if (bb.end.x >= img->xsize) {
    int offset = bb.end.x - img->xsize + 1;
    bb.begin.x -= offset;
    bb.end.x -= offset;
  }
  if (bb.end.y >= img->ysize) {
    int offset = bb.end.y - img->ysize + 1;
    bb.begin.y -= offset;
    bb.end.y -= offset;
  }

  return bb;
}

bool iftIsValidLabelMapBoundingBox(iftBoundingBox bb,   iftImage *labelMap,   iftImage *gt, int label, int *group)
{
  *group = BG_LABEL;
  bool hasRegionOverlap = false;
  for (iftVoxel u = bb.begin; u.x <= bb.end.x; ++(u.x)) {
    for (u.y = bb.begin.y; u.y <= bb.end.y; ++(u.y)) {
      int p = iftGetVoxelIndex(labelMap, u);
      if (gt->val[p] > 1)
        *group = OBJ_LABEL;
      if (labelMap->val[p] == label)
        hasRegionOverlap = true;
    }
  }

  return (hasRegionOverlap);
}

int main(int argc, char** argv) {

  if (argc < 6) {
    printf("Usage: %s <sample csv> <orig folder> <seg folder> <gt folder> <res folder>\n", argv[0]);
    exit(-1);
  }

  iftMakeDir(argv[5]);
  char strBuf[256];
  sprintf(strBuf, "%s/patch_info.csv", argv[5]);
  FILE *fp = fopen(strBuf, "w");
  fprintf(stdout, "path,orig,hasOverlap,x0,y0\n");
  fprintf(fp, "path,orig,hasOverlap,x0,y0\n");

  iftFileSet *fs = iftLoadFileSetFromDirOrCSV(argv[1], 1, true);
  int globalID = 1;

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

    printf("Extracting patches from label map...\n");

    iftMatrix *spCenters = iftComputeSuperpixelFeaturesByGeometricCenter(labelMap);

    for (int row = 0; row < spCenters->nrows; ++row) {
      int x = (int) roundf(iftMatrixElem(spCenters, 0, row));
      int y = (int) roundf(iftMatrixElem(spCenters, 1, row));
      iftBoundingBox bb = iftGetBoundingBoxFromCenter(x, y, img);

      int label = BG_LABEL;
      if (!iftIsValidLabelMapBoundingBox(bb, labelMap, gt, row + 1, &label))
        continue;

      iftImage *roi = iftExtractROI(img, bb);
      sprintf(strBuf, "%06d_%06d_%06d.png", label, globalID++, f+1);
      iftWriteImageByExt(roi, "%s/%s", argv[5], strBuf);
      fprintf(fp, "%s,%s,%d,%d,%d\n", strBuf, origPath, 2-label, bb.begin.x, bb.begin.y);

      iftDestroyImage(&roi);
    }

    free(csvPath);
    free(basename);
    free(origPath);
    free(segPath);
    free(gtPath);
    iftDestroyImage(&img);
    iftDestroyImage(&labelMap);
    iftDestroyImage(&gt);
    iftDestroyMatrix(&spCenters);
  }

  //iftDestroyFileSet(&fs);
  fclose(fp);

  return 0;
}

