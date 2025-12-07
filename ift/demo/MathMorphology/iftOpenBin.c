//
// Created by azael on 22/11/18.
//

#include "ift.h"

int main(int argc, const char *argv[])
{
    if (argc != 4)
        iftError("Usage: iftOpenBin <input image> <radius> <output image>","main");

    iftImage *img = iftReadImageByExt(argv[1]);
    float radius = atof(argv[2]);

    iftImage *out = iftOpenBin(img,radius);
    iftDestroyImage(&img);

    iftWriteImageByExt(out,argv[3]);
    iftDestroyImage(&out);

    return 0;
}

