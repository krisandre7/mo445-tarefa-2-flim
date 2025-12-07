#include "ift.h"


int main(int argc, const char* argv[]) {
    if (argc != 2)
        iftError("%s <color image>", "main", argv[0]);
    
    iftImage* img = iftReadImageByExt(argv[1]);

    iftImage* red   = iftImageRed(img);
    iftImage* green = iftImageGreen(img);
    iftImage* blue  = iftImageBlue(img);
    iftImage* Cb    = iftImageCb(img);
    iftImage* Cr    = iftImageCr(img);
    iftImage* Lum   = iftLuminance(img);
    
    iftWriteImageByExt(red,  "red.png");
    iftWriteImageByExt(green, "green.png");
    iftWriteImageByExt(blue, "blue.png");
    iftWriteImageByExt(Lum,  "Lum.png");
    iftWriteImageByExt(Cb, "Cb.png");
    iftWriteImageByExt(Cr, "Cr.png");

    iftDestroyImage(&red);
    iftDestroyImage(&green);
    iftDestroyImage(&blue);
    iftDestroyImage(&Cr);
    iftDestroyImage(&Cb);
    iftDestroyImage(&Lum);
    iftDestroyImage(&img);
    
    return 0;
}

