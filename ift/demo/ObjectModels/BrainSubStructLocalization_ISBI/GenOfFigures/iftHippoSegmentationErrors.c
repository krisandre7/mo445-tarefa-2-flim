#include "ift.h"






/**
 * 
 */
int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("iftHippoSegmentationErrors <segmentation> <gt> <output_errors: segmentation - gt>", "main");


    const char *seg_path = argv[1];
    const char *gt_path  = argv[2];
    const char *out_path = argv[3];

    printf("- Segmentation: %s\n", seg_path);
    printf("- Ground Truth: %s\n", gt_path);
    printf("- Output: %s\n\n", out_path);


    iftImage *seg = iftReadImageByExt(seg_path);
    iftImage *gt  = iftReadImageByExt(gt_path);

    iftImage *errors = iftXor(seg, gt);

    iftWriteImageByExt(errors, out_path);
    puts("Done...");

    iftDestroyImage(&seg);
    iftDestroyImage(&gt);
    iftDestroyImage(&errors);

    return 0;
}






