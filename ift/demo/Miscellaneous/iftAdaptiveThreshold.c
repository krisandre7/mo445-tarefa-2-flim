#include "ift.h"

int main(int argc, char **argv) {
    if (argc != 7)
        iftError("Please provide the following parameters:\n<input image> <output image> <output value (e.g., 255)> <perc (e.g., 0.98)> <niters (e.g., 2)> <adj. radius (e.g., 3.0)>\n\n", "main");

    char *filenameIn    = argv[1];
    char *filenameOut   = argv[2];
    int value           = atoi(argv[3]);
    float perc          = atof(argv[4]);
    int niters          = atoi(argv[5]);
    float radius        = atof(argv[6]);
    iftAdjRel *A        = NULL;
    
    iftImage* img = iftReadImageByExt(filenameIn);

    if (iftIs3DImage(img))
      A = iftSpheric(radius);
    else
      A = iftCircular(radius);
    
    iftImage *imgout = iftAboveAdaptiveThreshold(img,NULL,A,perc,niters,value);
  
    iftWriteImageByExt(imgout,filenameOut);

    iftDestroyImage(&img);
    iftDestroyImage(&imgout);
    iftDestroyAdjRel(&A);
    
    return 0;
}
