/**
 * @file
 * @brief Demo to check importance of small superpixels.
 * @author Felipe Lemes Galvao
 */

#include <ift.h>
#include "iftExperimentUtility.h"

int main(int argc, char* argv[])
{
  if (argc < 5) {
    printf("Usage: iftFindSmallSuperpixels <img> <seg> <thresh> <res>\n");
    return -1;
  }

  // Load args
  iftImage *img = iftReadImageByExt(argv[1]);
  iftImage *labelMap = iftReadImageByExt(argv[2]);
  float sizeThreshold = atof(argv[3]);

  // Init
  int nSuperpixels = iftMaximumValue(labelMap);
  int *spSize = iftAllocIntArray(nSuperpixels);

  // Find each superpixel size
  for (int p = 0; p < labelMap->n; ++p) {
    int label = labelMap->val[p] - 1;
    spSize[label] += 1;
  }

  // Paint borders of superpixels below threshold
  int count = 0;
  iftColorTable *colorTb = iftCreateColorTable(1);
  float meanSpSize = (float)labelMap->n / (float)nSuperpixels;
  for (int s = 0; s < nSuperpixels; ++s) {
    if (spSize[s] >= sizeThreshold * meanSpSize)
      continue;

    iftDrawBordersSingleLabel(img, labelMap, s+1, colorTb->color[0]);
    count += 1;
  }

  // Save result
  printf("Average superpixel size = %f (%d/%d)\n", meanSpSize, labelMap->n, nSuperpixels);
  printf("Filtered superpixels with size less than %f\n", sizeThreshold * meanSpSize);
  printf("Found %d out of %d superpixels.\n", count, nSuperpixels);
  iftWriteImageByExt(img, argv[4]);

  return 0;
}
