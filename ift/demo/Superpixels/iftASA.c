#include "ift.h"

// try:
// iftASA synthetic-test-cases/test-case-1/fake-superpixels-1.pgm synthetic-test-cases/test-case-1/gt.pgm


int main(int argc, const char* argv[]) {
    if (argc != 3)
        iftError("%s <superpixel-map> <ground-truth-label-image>", "main", argv[0]);
    
    
    iftImage* super_map = iftReadImageByExt(argv[1]);
    iftImage* gt = iftReadImageByExt(argv[2]);
    
    double asa = iftASA(super_map, gt);
    printf("Achievable segmentation accuracy (ASA) = %lf\n", asa);
    
    return 0;
}
