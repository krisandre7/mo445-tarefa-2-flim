#include "ift.h"




int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("iftCloseBin <input_img> <radius> <out_img>", "main");

    iftImage *bin = iftReadImageByExt(argv[1]);
    float radius = atof(argv[2]);

    puts("- Closing Bin Image");
    iftImage *out = iftCloseBin(bin, radius);
    iftWriteImageByExt(out, argv[3]);

    iftDestroyImage(&bin);
    iftDestroyImage(&out);

    return 0;
}
