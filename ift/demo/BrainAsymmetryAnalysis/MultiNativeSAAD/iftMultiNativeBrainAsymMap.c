#include "ift.h"


int main(int argc, const char *argv[]) {
    if (argc != 5)
        iftError("%s <image> <template_reg> <mask> <out_asym_map", "main", argv[0]);
    
    iftImage *img = iftReadImageByExt(argv[1]);
    iftImage *template_img = iftReadImageByExt(argv[2]);
    iftImage *mask = iftReadImageByExt(argv[3]);
    
    iftImage *img_mask = iftMask(img, mask);
    iftImage *template_img_mask = iftMask(template_img, mask);
    iftImage *template_img_hist = iftMatchHistogram(template_img_mask, NULL, img_mask, NULL);
    iftWriteImageByExt(template_img_hist, "tmp/template_img_hist.nii.gz");
    
    iftImage *asym_map = iftAbsSub(img_mask, template_img_hist);
    
    iftWriteImageByExt(asym_map, argv[4]);
    
    return 0;
}

