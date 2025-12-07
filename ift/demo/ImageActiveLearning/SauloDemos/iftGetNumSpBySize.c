#include "ift.h"
#include "iftExperimentUtility.h"

int main(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: %s <label map> <max size>\n", argv[0]);
    return -1;
  }

  iftImage *labelMap = iftReadImageByExt(argv[1]);
  int maxSize = atol(argv[2]);

  // Get Sizes
  int nSp = iftMaximumValue(labelMap);
  int *sizes = iftGetSuperpixelSizes(labelMap, nSp); 
  int countedSize = 0;
  int count = 0;
  for (int i = 0; i < nSp; ++i) {
    if (sizes[i] <= maxSize) {
      ++count;
      countedSize += sizes[i];
    }
  }

  printf("%d out of %d superpixels have size <= %d.\n", count, nSp, maxSize);
  printf("They total %d out of %d pixels.\n", countedSize, labelMap->n);
  iftDestroyImage(&labelMap);
  free(sizes);

  return 0;
}
