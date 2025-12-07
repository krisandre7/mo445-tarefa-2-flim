#include "ift.h"
#include "iftExperimentUtility.h"

int main(int argc, char* argv[])
{
  if (argc < 4) {
    printf("Usage: %s <label map> <ground truth> <acceptable fraction> <output binary map>\n", argv[0]);
    return -1;
  }

  iftImage *labelMap = iftReadImageByExt(argv[1]);
  iftImage *gt = iftReadImageByExt(argv[2]);
  float threshold = atof(argv[3]);
  iftImage *res = iftCopyImage(labelMap);

  float *overlap = iftGetSuperpixelObjectOverlap(labelMap, gt, -1);
  for (int p = 0; p < labelMap->n; ++p) {
    int label = labelMap->val[p] - 1;
    if (label < 0)
      continue;

    res->val[p] = (overlap[label] >= threshold ? 255 : 0);
  }

  iftWriteImageByExt(res, argv[4]);

  iftDestroyImage(&labelMap);
  iftDestroyImage(&gt);
  iftDestroyImage(&res);
  free(overlap);

  return 0;
}
