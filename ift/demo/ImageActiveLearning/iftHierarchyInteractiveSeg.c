#include "ift.h"

// Assumption: binary segmentation

#define VISUALIZATION_FLAG 1
#define N_ITERS 50
#define ERROR_TOLERANCE 0

iftImage * iftTrimSpBorders(iftImage *sp, int minDistSqr);
int * iftFindSpSize(  iftImage *sp, int nSp);
iftSet * iftFindCandidateSp(  iftImage *sp, int *spSize, int nSp,   iftImage *gt);
iftImage * iftWatershedFromSeedImage(  iftImage *basins,   iftImage *seeds,   iftImage *gt);
iftImage * iftSegErrorImage(  iftImage *seg,   iftImage *gt);
void iftUpdateSpErrors(  iftImage *segError,   iftImage *sp, int *spError, int nSp);
int iftRobotMarkSp(int nScales, iftSet **candidateArr, int **spErrorArr, iftImage **spArr, iftImage *seeds, iftImage *gt);

int main(int argc, char* argv[])
{
  if (argc < 6) {
    fprintf(stderr, "Usage %s <baseImg> <gtLabelMap> <resFolder> <spScale1(coarsest)> [spScale2 (...)]\n", argv[0]);
    return -1;
  }

  iftImage *img = iftReadImageByExt(argv[1]);
  iftImage *gt = iftReadImageByExt(argv[2]);
  int nScales = argc - 4;
  iftImage **spArr = malloc(nScales * sizeof(*spArr));
  for (int i = 0; i < nScales; ++i)
    spArr[i] = iftReadImageByExt(argv[4+i]);

  for (int i = 0; i < nScales; ++i) {
    int minDistSqr = nScales - i;
    minDistSqr *= minDistSqr;
    minDistSqr /= 2;

    iftImage *tmp = iftTrimSpBorders(spArr[i], minDistSqr);
    iftDestroyImage(&(spArr[i]));
    spArr[i] = tmp;
  }
  
  // Normalize ground truth
  for (int i = 0; i < gt->n; ++i)
    if (gt->val[i] != 0)
      gt->val[i] = 255;

  int *nSpArr = malloc(nScales * sizeof(*nSpArr));
  int **spSizeArr = malloc(nScales * sizeof(*spSizeArr));
  for (int i = 0; i < nScales; ++i) {
    nSpArr[i] = iftMaximumValue(spArr[i]);
    spSizeArr[i] = iftFindSpSize(spArr[i], nSpArr[i]);
  }

  iftSet **candidateArr = malloc(nScales * sizeof(*candidateArr));
  for (int i = 0; i < nScales; ++i)
    candidateArr[i] = iftFindCandidateSp(spArr[i], spSizeArr[i], nSpArr[i], gt);

  // Visualization of usable superpixels
  if (VISUALIZATION_FLAG) {
    for (int i = 0; i < nScales; ++i) {
      iftImage *vis = iftCreateImage(img->xsize, img->ysize, img->zsize);

      int *candidateFlag = iftAllocIntArray(nSpArr[i]);
      for (iftSet *s = candidateArr[i]; s != NULL; s = s->next)
        candidateFlag[s->elem] = 1;

      for (int p = 0; p < vis->n; ++p) {
        int label = spArr[i]->val[p] - 1;
        if (candidateFlag[label])
          vis->val[p] = 255;
      }

      iftWriteImageByExt(vis, "%s/usable_scale_%02d.pgm", argv[3], i);

      free(candidateFlag);
      iftDestroyImage(&vis);
    }
  }

  iftImage *seeds = NULL;
  iftImage *seg = NULL;
  iftImage *basins = iftImageBasins(img, NULL);
  int **spErrorArr = malloc(nScales * sizeof(*spErrorArr));
  for (int i = 0; i < nScales; ++i)
    spErrorArr[i] = iftAllocIntArray(nSpArr[i]);
  double *diceArr = iftAllocDoubleArray(N_ITERS);
  int nDice = 0;

  // Interactive loop with robot
  for (int iter = 0; iter < N_ITERS; ++iter) {
    if (seeds == NULL) {
      seeds = iftCreateImage(img->xsize, img->ysize, img->zsize);
      seg = iftCreateImage(img->xsize, img->ysize, img->zsize);
      for (int i = 0; i < seeds->n; ++i) {
        seeds->val[i] = -1;
        seg->val[i] = -1;
      }
    } else {
      iftImage *tmp = iftWatershedFromSeedImage(basins, seeds, gt);
      iftDestroyImage(&seg);
      seg = tmp;

      // Seeds visualization
      if (VISUALIZATION_FLAG) {
        iftImage *vis = iftCreateImage(img->xsize, img->ysize, img->zsize);
        for (int i = 0; i < img->n; ++i)
          vis->val[i] = seeds->val[i] < 0 ? 127 : seeds->val[i] == 0 ? 0 : 255;
        iftWriteImageByExt(vis, "%s/seeds_%06d.pgm", argv[3], iter);
        iftDestroyImage(&vis);
      }

      // Segmentation visualization
      if (VISUALIZATION_FLAG) {
        iftImage *vis = iftCreateImage(img->xsize, img->ysize, img->zsize);
        for (int i = 0; i < img->n; ++i)
          vis->val[i] = seg->val[i] > 0 ? 255 : 0;
        iftWriteImageByExt(vis, "%s/seg_%06d.pgm", argv[3], iter);
        iftDestroyImage(&vis);
      }
    }

    iftImage *segError = iftSegErrorImage(seg, gt);


    if (VISUALIZATION_FLAG) {
      iftImage *vis = iftCreateImage(img->xsize, img->ysize, img->zsize);
      for (int i = 0; i < img->n; ++i)
        vis->val[i] = segError->val[i] * 255;
      iftWriteImageByExt(vis, "%s/error_%06d.pgm", argv[3], iter);
      iftDestroyImage(&vis);
    }

    for (int i = 0; i < nScales; ++i)
      iftUpdateSpErrors(segError, spArr[i], spErrorArr[i], nSpArr[i]);

    int status = iftRobotMarkSp(nScales, candidateArr, spErrorArr, spArr, seeds, gt);
    if (status == 0)
      break;

    // Stats
    diceArr[nDice++] = iftDiceSimilarity(seg, gt);
  }

  iftWriteImageByExt(seg, "%s/res.pgm", argv[3]);
  printf("%f", diceArr[0]);
  for (int i = 1; i < nDice; ++i)
    printf(", %f", diceArr[i]);
  for (int i = nDice; i < N_ITERS; ++i)
    printf(", %f", diceArr[nDice-1]);

  return 0;
}

iftImage * iftTrimSpBorders(iftImage *sp, int minDistSqr)
{
  // Ignore pixels at a certain distance from sp borders
  iftAdjRel *A = iftCircular(1.0);
  iftImage *borderDistSqr = iftBorderDistTrans(sp, A);

  iftImage *res = iftCopyImage(sp);
  for (int i = 0; i < sp->n; ++i)
    if (borderDistSqr->val[i] < minDistSqr)
      res->val[i] = 0;

  iftDestroyAdjRel(&A);
  iftDestroyImage(&borderDistSqr);

  return res;
}

int * iftFindSpSize(  iftImage *sp, int nSp)
{
  int *spSize = iftAllocIntArray(nSp);

  for (int i = 0; i < sp->n; ++i) {
    int s = sp->val[i] - 1;
    if (s < 0)
      continue;
    spSize[s] += 1;
  }

  return spSize;
}

iftSet * iftFindCandidateSp(  iftImage *sp, int *spSize, int nSp,   iftImage *gt)
{
  iftSet *res = NULL;

  int *spLabel = iftAllocIntArray(nSp);
  for (int i = 0; i < sp->n; ++i) {
    int s = sp->val[i] - 1;
    if (s < 0)
      continue;
    spLabel[s] += (gt->val[i] != 0) ? 1 : -1;
  }

  for (int s = 0; s < nSp; ++s) {
    if (spSize[s] == 0)
      continue;
    spLabel[s] = spSize[s] - abs(spLabel[s]);
    float error = (float)spLabel[s]/(float)spSize[s];
    // For now we assume the tolerance from the trimmed sps are enough
    if (error < IFT_EPSILON)
      iftInsertSet(&res, s);
  }

  free(spLabel);
  return res;
}

iftImage * iftWatershedFromSeedImage(  iftImage *basins,   iftImage *seeds,   iftImage *gt)
{
  iftLabeledSet *seedSet = NULL;
  for (int i = 0; i < seeds->n; ++i) {
    if (seeds->val[i] < 0)
      continue;

    iftInsertLabeledSet(&seedSet, i, seeds->val[i]);
  }

  iftImage *seg = iftWatershed(basins, NULL, seedSet, NULL);

  iftDestroyLabeledSet(&seedSet);
  return seg;
}

iftImage * iftSegErrorImage(  iftImage *seg,   iftImage *gt)
{
  iftImage *segError = iftCreateImage(seg->xsize, seg->ysize, seg->zsize);
  for (int i = 0; i < seg->n; ++i)
    segError->val[i] = (seg->val[i] != gt->val[i]) ? 1 : 0;
  
  return segError;
}

void iftUpdateSpErrors(  iftImage *segError,   iftImage *sp, int *spError, int nSp)
{
  for (int i = 0; i < nSp; ++i)
    spError[i] = 0;

  for (int i = 0; i < sp->n; ++i) {
    int s = sp->val[i] - 1;
    if (s < 0)
      continue;
    spError[s] += segError->val[i];
  }
}

// Returns true if seeds were actually added
int iftRobotMarkSp(int nScales, iftSet **candidateArr, int **spErrorArr, iftImage **spArr, iftImage *seeds, iftImage *gt)
{
  int markerScale;
  int markerSp;
  int markerSpErrors = 0;

  // Find sp with most errors across all scales
  for (int i = 0; i < nScales; ++i) {
    for (iftSet *S = candidateArr[i]; S != NULL; S = S->next) {
      int nErrors = spErrorArr[i][S->elem];
      if (nErrors > markerSpErrors) {
        markerScale = i;
        markerSp = S->elem;
        markerSpErrors = nErrors;
      }
    }
  }

  if (markerSpErrors <= ERROR_TOLERANCE)
    return 0;

  // Update seeds
  int label = -1;
  for (int i = 0; i < seeds->n; ++i) {
    int s = spArr[markerScale]->val[i] - 1;
    if (s != markerSp)
      continue;

    // Make sure we are not cheating when using gt over sp as
    //   marker should have an unique label
    if (label < 0) {
      label = gt->val[i];
    }
    // Should always be true by current definition of candidate list
    assert(label == gt->val[i]);

    seeds->val[i] = label;
  }

  //printf("Adding seeds with label %d on scale %d, sp %d with %d errors.\n", label, markerScale, markerSp, markerSpErrors);

  return 1;
}
