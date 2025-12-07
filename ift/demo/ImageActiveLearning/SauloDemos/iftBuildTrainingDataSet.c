#include "ift.h"
#include "iftExperimentUtility.h"
#include "iftSuperpixelFeatures.h"

static   int TARGET_RADIUS = 49;
static   int BG_LABEL = 2;
static   int OBJ_LABEL = 1;
static   int MAX_CANDIDATE_SIZE = 3000; // should match iftSizeFilteringMerge

iftAdjRel *iftCreateOffsetAdjRel()
{
    static int NUM_ANGLES = 4;
    static int NUM_DIST = 3;
    static int MAX_DIST = 30;
    static int QUARTER_N = NUM_ANGLES * NUM_DIST;
  int n = 1 + QUARTER_N * 4;
  iftAdjRel *offset = iftCreateAdjRel(n);

  // Fill all positions within a 90 degree space
  for (int i = 0; i < NUM_DIST; ++i) {
    float radiusStep = ((float) MAX_DIST) /  ((float) NUM_DIST);
    float radius = radiusStep * ((float) (i + 1));
    for (int j = 0; j < NUM_ANGLES; ++j) {
      float angleStep = (IFT_PI / 2.0) / ((float) NUM_ANGLES);
      float angle = angleStep * ((float) j);
      int idx = i * NUM_ANGLES + j + 1;
      offset->dx[idx] = roundf(cosf(angle) * radius);
      offset->dy[idx] = roundf(sinf(angle) * radius);
    }
  }

  // Fill rotations for the full circle
  for (int i = 1 + QUARTER_N; i < 1 + 2 * QUARTER_N; ++i) {
    offset->dx[i] = - offset->dy[i - QUARTER_N];
    offset->dy[i] = offset->dx[i - QUARTER_N];
  }
  for (int i = 1 + 2 * QUARTER_N; i < 1 + 4 * QUARTER_N; ++i) {
    offset->dx[i] = - offset->dx[i - 2 * QUARTER_N];
    offset->dy[i] = - offset->dy[i - 2 * QUARTER_N];
  }

  return offset;
}

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

bool iftIsValidLabelMapBoundingBox(iftBoundingBox bb,   iftImage *labelMap,   iftImage *gt, int label)
{
  bool hasGtOverlap = false;
  bool hasRegionOverlap = false;
  for (iftVoxel u = bb.begin; u.x <= bb.end.x; ++(u.x)) {
    for (u.y = bb.begin.y; u.y <= bb.end.y; ++(u.y)) {
      int p = iftGetVoxelIndex(labelMap, u);
      if (gt->val[p] > 1)
        hasGtOverlap = true;
      if (labelMap->val[p] == label)
        hasRegionOverlap = true;
    }
  }
  return (!hasGtOverlap && hasRegionOverlap);
}

int main(int argc, char** argv) {

  if (argc < 6) {
    printf("Usage: %s <sample csv> <orig folder> <seg folder> <gt folder> <res folder>\n", argv[0]);
    exit(-1);
  }

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

    printf("Extracting patches from ground truth...\n");

    // Label gt connected components
    iftImage *tmpImg = iftForceLabelMapConnectivity(gt, 0);
    iftDestroyImage(&gt);
    gt = tmpImg;

    iftMatrix *gtObjCenters = iftComputeSuperpixelFeaturesByGeometricCenter(gt);
    iftAdjRel *gtOffset = iftCreateOffsetAdjRel();

      static int firstNonBGRow = 1;
    for (int row = firstNonBGRow; row < gtObjCenters->nrows; ++row) {
      int xCenter = (int) roundf(iftMatrixElem(gtObjCenters, 0, row));
      int yCenter = (int) roundf(iftMatrixElem(gtObjCenters, 1, row));
      for (int i = 0; i < gtOffset->n; ++i) {
        int x = xCenter + gtOffset->dx[i];
        int y = yCenter + gtOffset->dy[i];
        iftBoundingBox bb = iftGetBoundingBoxFromCenter(x, y, img);
        iftImage *roi = iftExtractROI(img, bb);
        iftWriteImageByExt(roi, "%s/%06d_%06d_%06d.png", argv[5], OBJ_LABEL, globalID++, f+1);
        iftDestroyImage(&roi);
      }
    }

    printf("Extracting patches from label map...\n");

    iftMatrix *spCenters = iftComputeSuperpixelFeaturesByGeometricCenter(labelMap);

    for (int row = 0; row < spCenters->nrows; ++row) {
      int x = (int) roundf(iftMatrixElem(spCenters, 0, row));
      int y = (int) roundf(iftMatrixElem(spCenters, 1, row));
      iftBoundingBox bb = iftGetBoundingBoxFromCenter(x, y, img);

      if (!iftIsValidLabelMapBoundingBox(bb, labelMap, gt, row + 1))
        continue;

      iftImage *roi = iftExtractROI(img, bb);
      iftWriteImageByExt(roi, "%s/%06d_%06d_%06d.png", argv[5], BG_LABEL, globalID++, f+1);
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
    iftDestroyMatrix(&gtObjCenters);
    iftDestroyAdjRel(&gtOffset);
    iftDestroyMatrix(&spCenters);
  }

  //iftDestroyFileSet(&fs);

  return 0;
}

