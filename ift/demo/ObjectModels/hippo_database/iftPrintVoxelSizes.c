#include "ift.h"





int main(int argc, const char *argv[]) {
    if (argc != 2)
        iftError("iftPrintVoxelSizes <image>", "main");

    iftImage *img = iftReadImageByExt(argv[1]);
    puts(argv[1]);
    printf("voxel sizes: (%f, %f, %f)\n\n", img->dx, img->dy, img->dz);

    return 0;
}










