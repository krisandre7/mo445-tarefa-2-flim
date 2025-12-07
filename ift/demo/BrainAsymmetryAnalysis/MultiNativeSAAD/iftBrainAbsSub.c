#include "ift.h"


int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("%s <first_image> <second_image> <out_abs_sub>", "main", argv[0]);
    
    iftImage *img1 = iftReadImageByExt(argv[1]);
    iftImage *img2 = iftReadImageByExt(argv[2]);
    iftWriteImageByExt(iftAbsSub(img1, img2), argv[3]);
    
    return 0;
}
