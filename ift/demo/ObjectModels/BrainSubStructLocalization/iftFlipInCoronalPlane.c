#include "ift.h"



int main(int argc, const char *argv[]) {
    if (argc != 3)
        iftError("iftFlipInCoronalSlice <image> <output_image>", "main");

    char *img_path = iftCopyString(argv[1]);
    char *out_img_path = iftCopyString(argv[2]);


    iftImage *img = iftReadImageByExt(img_path);
    printf("Flipping in Coronal Plane: %s --> %s\n", img_path, out_img_path);
    iftImage *flipped = iftFlipImage(img, IFT_AXIS_X);
    iftWriteImageByExt(flipped, out_img_path);

    return 0;
}


