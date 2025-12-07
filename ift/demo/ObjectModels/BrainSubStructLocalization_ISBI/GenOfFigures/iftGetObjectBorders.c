#include "ift.h"



int main(int argc, const char *argv[]) {
    if (argc != 3)
        iftError("iftGetObjectBorders <label_img> <output_border_img>", "main");

    iftImage *label_img = iftReadImageByExt(argv[1]);
    iftImage *borders = iftObjectBorders(label_img, NULL, true, true);

    iftWriteImageByExt(borders, argv[2]);
    puts("Done...");

    return 0;
}




