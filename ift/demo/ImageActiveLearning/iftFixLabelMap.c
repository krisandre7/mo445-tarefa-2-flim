#include "ift.h"

extern void iftRemoveLabelGaps(int *labelArray, int size);

int main(int argc, char* argv[])
{
  iftImage *img = iftReadImageByExt(argv[1]);
  if (iftMinimumValue(img) == 0)
    for (int i = 0; i < img->n; ++i)
      img->val[i] += 1;
  iftRemoveLabelGaps(img->val, img->n);
  iftWriteImageByExt(img, argv[1]);

  iftDestroyImage(&img);

  return 0;
}
