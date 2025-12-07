#include "ift.h"



double iftIoU(  iftImage* super_map,   iftImage* gt) {
    iftVerifyImageDomains(super_map, gt, "iftIoU");
    
    int min_superpixel;
    int n_superpixels;
    iftMinMaxValues(super_map, &min_superpixel, &n_superpixels);
    
    if (min_superpixel <= 0)
        iftError("Superpixel Map must only have values >= 1.\nFound: %d", "iftIoU", min_superpixel);
    
    int min_obj;
    int n_objs;
    iftMinMaxValues(gt, &min_obj, &n_objs);
    
    if (min_obj < 0)
        iftError("Ground-Truth Label Image must only have values >= 0, where 0 is the background.\nFound: %d",
                 "iftIoU", min_obj);
}




int main(int argc, const char* argv[]) {
    if (argc != 3)
        iftError("%s <superpixel-map> <ground-truth-label-image>", "main", argv[0]);
    
    char* super_img_path = iftCopyString(argv[1]);
    char* gt_path = iftCopyString(argv[1]);
    
    puts("----------------------");
    printf("- Superpixel Path: %s\n", super_img_path);
    printf("- Ground-Truth Label Image: %s\n", gt_path);
    puts("----------------------");
    
    iftImage* super_img = iftReadImageByExt(super_img_path);
    iftImage* gt = iftReadImageByExt(gt_path);
    
    
    
    return 0;
}
