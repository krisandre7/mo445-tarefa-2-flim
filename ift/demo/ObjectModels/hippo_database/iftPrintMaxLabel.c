#include "ift.h"



int main(int argc, const char *argv[]) {
    if (argc != 2)
        iftError("iftPrintMaxLabel <label_img>", "main");

    iftImage *label_img = iftReadImageByExt(argv[1]);
    printf("max label = %d\n", iftMaximumValue(label_img));

    return 0;
}


