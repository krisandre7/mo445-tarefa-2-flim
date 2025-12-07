#include "ift.h"


int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("%s <bin_mask> <dilation_radius> <out_dilated_bin_mask>", "main", argv[0]);

    iftImage *bin_mask = iftReadImageByExt(argv[1]);
    iftSet *S = NULL;
    iftImage *bin_mask_dilated = iftDilateBin(bin_mask, &S, atof(argv[2]));
    iftWriteImageByExt(bin_mask_dilated, argv[3]);

    return 0;
}
