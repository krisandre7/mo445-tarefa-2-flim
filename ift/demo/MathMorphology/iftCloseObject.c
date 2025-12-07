#include "ift.h"


int main(int argc, const char *argv[]) {
    if (argc != 5)
        iftError("iftCloseBin <input_img> <label> <radius> <out_img>", "main");

    iftImage *label_img = iftReadImageByExt(argv[1]);
    int obj_label = atoi(argv[2]);
    float radius = atof(argv[3]);

    iftImage *obj_img = iftExtractObject(label_img, obj_label);
    iftImage *out = iftCloseBin(obj_img, radius);
    
    for (int p = 0; p < label_img->n; p++) {
        if (label_img->val[p] && label_img->val[p] != obj_label) {
            out->val[p] = label_img->val[p];
        }
    }
    
    iftWriteImageByExt(out, argv[4]);


    return 0;
}
