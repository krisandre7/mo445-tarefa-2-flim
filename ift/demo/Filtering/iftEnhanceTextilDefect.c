#include "ift.h"


int main(int argc, const char *argv[]) {
    if (argc != 3)
      iftError("iftEnhanceTextilDefect <input_image> <output_image>", "main");

    /* Attenuate the regular pattern of the fabric while preserving
       the defects */
    
    iftImage  *img      = iftReadImageByExt(argv[1]);
    iftAdjRel *A        = iftRectangular(5,5);
    iftImage *filt_img  = iftMedianFilter(img, A);
    iftImage *leveling  = iftLeveling(img,filt_img);
    iftImage  *bin      = iftThreshold(leveling,0.90*iftOtsu(leveling),255,255);
    
    iftWriteImageByExt(bin, argv[2]);

    iftDestroyAdjRel(&A);
    iftDestroyImage(&img);
    iftDestroyImage(&filt_img);
    iftDestroyImage(&bin);
    iftDestroyImage(&leveling);
    
    return 0;
}






