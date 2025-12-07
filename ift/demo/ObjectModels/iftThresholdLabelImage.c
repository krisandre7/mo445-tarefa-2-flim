#include "ift.h"



int main(int argc, const char *argv[]) {
    if (argc != 5)
        iftError("program <mri> <label> <threshold> <out_image>", "main");

    iftImage *img = iftReadImageByExt(argv[1]);
    iftImage *label_img = iftReadImageByExt(argv[2]);
    int lower_thres = atoi(argv[3]);

    iftImage *threshold_img = iftThreshold(img, lower_thres, 4095, 1);
    iftImage *out = iftMask(label_img, threshold_img);
    iftWriteImageByExt(out, argv[4]);


    return 0;
}

