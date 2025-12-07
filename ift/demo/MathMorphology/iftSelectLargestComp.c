//
// Created by azael on 22/11/18.
//

#include "ift.h"

int main(int argc, const char *argv[])
{
    if (argc != 4)
        iftError("Usage: iftSelectLargestComp <input image> <adj. radius> <output image>","main");

    iftImage *img = iftReadImageByExt(argv[1]);
    float radius = atof(argv[2]);
    iftAdjRel *A = NULL;
    
    if (iftIs3DImage(img))
      A = iftSpheric(radius);
    else
      A = iftCircular(radius);
    
    iftImage *out = iftSelectLargestComp(img,A);

    iftDestroyImage(&img);
    iftWriteImageByExt(out,argv[3]);
    iftDestroyImage(&out);
    iftDestroyAdjRel(&A);
    
    return 0;
}

