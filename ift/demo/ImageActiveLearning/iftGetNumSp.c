#include "ift.h"

int main(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: %s <input_img> [empty if label_img or any value for border image]\n", argv[0]);
    return -1;
  }

  iftImage *img = iftReadImageByExt(argv[1]);

  if (argc == 2) {
    printf("%d\n", iftMaximumValue(img));
  } else {
    // Input img is lost in this function
    iftImage *labelMap = iftBorderImageToLabelImage(img);
    printf("%d\n", iftMaximumValue(labelMap));
    iftDestroyImage(&labelMap);
  }

  iftDestroyImage(&img);

  return 0;
}
