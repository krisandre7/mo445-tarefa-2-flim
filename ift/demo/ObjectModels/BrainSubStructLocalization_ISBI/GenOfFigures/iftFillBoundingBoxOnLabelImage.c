#include "ift.h"




int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("iftFillBoundingBoxOnLabelImage <label_img> <bb.json> <output_img>", "main");


    iftImage *label_img            = iftReadImageByExt(argv[1]);
    iftLabelBoundingBoxArray *lbbs = iftReadLabelBoundingBoxArray(argv[2]);

    iftImage *bin_img = iftBinarize(label_img);

    iftImage *bb_img = iftCreateImageFromImage(bin_img);

    for (int o = 0; o < lbbs->labels->n; o++) {
        iftFillBoundingBoxInImage(bb_img, lbbs->bbs[o], 10);
    }

    iftWriteImageByExt(iftAdd(bb_img, bin_img), argv[3]);


    return 0;
}







