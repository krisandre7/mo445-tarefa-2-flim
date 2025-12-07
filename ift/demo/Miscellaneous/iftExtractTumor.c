#include "ift.h"

int main(int argc, char **argv) {
    if (argc != 3)
        iftError("Please provide the following parameters:\n<input image> <output image>\n\n", "main");

    char *filenameIn    = argv[1];
    char *filenameOut   = argv[2];
    iftImage* img = iftReadImageByExt(filenameIn);

    iftImage *brain_mask = iftThreshold(img,1,IFT_INFINITY_INT,255);
    iftSet   *S          = iftObjectBorderSet(brain_mask, NULL);
    // I was expecting that the brain border S with a threshold at 1
    // would work better than the image border (default) with Otsu's
    // threshold. However, it was the other way around.
    //iftImage *odomes     = iftOpenDomes(img,S,NULL);
    iftImage *odomes     = iftOpenDomes(img,NULL,NULL);
    iftImage *resid      = iftSub(img,odomes);
    iftAdjRel *A         = iftSpheric(1.0);
    iftImage *open       = iftOpen(resid,A,NULL);
    iftImage *tumor      = iftThreshold(open,iftOtsu(open),IFT_INFINITY_INT,255);
    iftWriteImageByExt(tumor,filenameOut);

    iftDestroyImage(&img);
    iftDestroyImage(&tumor);
    iftDestroyImage(&odomes);
    iftDestroyImage(&brain_mask);
    iftDestroyImage(&open);
    iftDestroyImage(&resid);
    iftDestroySet(&S);
    iftDestroyAdjRel(&A);
    
    return 0;
}
