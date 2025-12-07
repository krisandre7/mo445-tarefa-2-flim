#include "ift.h"

int main(int argc, char* argv[])
{
  iftImage *img = iftReadImageByExt(argv[1]);

  printf("P5\n");
  printf("%d %d\n", img->xsize, img->ysize);
  printf("%d\n", (1 << 16) - 1);

  iftDestroyImage(&img);

  return 0;
}
