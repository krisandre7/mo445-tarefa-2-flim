#include "ift.h"


int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("iftThresholdGT <input_img> <gt> <out_img>", "main");

    const char *img_path = argv[1];
    const char *gt_path  = argv[2];
    const char *out_path = argv[3];

    iftImage *img     = iftReadImageByExt(img_path);
    iftImage *gt      = iftReadImageByExt(gt_path);
    
    int thres = iftOtsu(img) * 1.5;
    printf("- Threshold: %d\n", thres);

    iftImage *new_gt = iftCreateImage(gt->xsize, gt->ysize, gt->zsize);
    for (int p = 0; p < new_gt->n; p++) {
        if (img->val[p] >= thres)
            new_gt->val[p] = gt->val[p];
    }

    iftWriteImageByExt(new_gt, out_path);

    iftDestroyImage(&gt);
    iftDestroyImage(&new_gt);

    return 0;
}