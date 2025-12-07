#include "ift.h"

void iftWriteSuperpixelBorders(iftImage *img,   iftImage *superpixels, const char *out_img_path,
                              int img_max_range) {
    const char *ext = iftFileExt(out_img_path);
    char *filename = iftFilename(out_img_path, ext);
    char *border_img_path = iftConcatStrings(3, filename, "_borders", ext);

    iftImage *border_img = iftBorderImage(superpixels);
    iftAdjRel *A         = iftSpheric(0.0);

    iftColor RGB;
    iftColor YCbCr;
    if (iftIs3DImage(superpixels)) {
        RGB.val[0] = RGB.val[1] = RGB.val[2] = img_max_range;
        YCbCr = iftRGBtoYCbCr(RGB, img_max_range);
    }
    else {
        RGB.val[0] = img_max_range;
        RGB.val[1] = RGB.val[2] = 0;
        YCbCr = iftRGBtoYCbCr(RGB, img_max_range);
    }

    // draws borders on the image
    iftDrawBorders(img, border_img, A, YCbCr, A);
    iftWriteImageByExt(img, border_img_path);

    // DESTROYERS
    iftDestroyAdjRel(&A);
    iftDestroyImage(&border_img);
    iftFree(filename);
    iftFree(border_img_path);
}


int main(int argc, const char *argv[]) {
    if (argc != 3)
        iftError("iftWriteSuperpixelBorders <img> <superpixels>", "main");

    iftImage *img = iftReadImageByExt(argv[1]);
    iftImage *super = iftReadImageByExt(argv[2]);

    iftWriteSuperpixelBorders(img, super, argv[2], 255);

    return 0;
}