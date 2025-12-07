#include "ift.h"


// this programs relabels a binary mask of hippocampus (binary image with a single or both hippocampus),
// so that the object to the left of the center x-coordinate is the left hippocampus and has label 1,
// and the object to the right of the center x-coordinate is the right hippo and has label 2.
// 
// If the image has only hone hippocampus, it will have label 1
int main(int argc, const char *argv[]) {
    if (argc != 3)
        iftError("iftRelabelHippocampus <input_binary_image> <output_binary_image>", "main");
    
    iftImage *bin_img = iftReadImageByExt(argv[1]);

    int x_center_coord = bin_img->xsize / 2;
    printf("x_center_coord = %d\n", x_center_coord);

    int left_hippo_label = 1;
    int right_hippo_label = 2;

    // bool left_hippo_exists = false;

    iftVoxel v;
    for (v.z = 0; v.z < bin_img->zsize; v.z++) {
        for (v.y = 0; v.y < bin_img->ysize; v.y++) {
            // label the left hippocampus
            for (v.x = 0; v.x < x_center_coord; v.x++) {
                if (iftImgVoxelVal(bin_img, v) != 0) {
                    iftImgVoxelVal(bin_img, v) = left_hippo_label;
                    // left_hippo_exists = true;
                }
            }
        }
    }

    // // if there is no left hippocampus, the label of the right hippo should be 1
    // if (!left_hippo_exists)
    //     right_hippo_label = 1;


    for (v.z = 0; v.z < bin_img->zsize; v.z++) {
        for (v.y = 0; v.y < bin_img->ysize; v.y++) {
            // label the right hippocampus
            for (v.x = x_center_coord; v.x < bin_img->xsize; v.x++) {
                if (iftImgVoxelVal(bin_img, v) != 0) {
                    iftImgVoxelVal(bin_img, v) = right_hippo_label;
                }
            }
        }
    }

    iftWriteImageByExt(bin_img, argv[2]);

    iftDestroyImage(&bin_img);

    puts("Done...");

    return 0;
}




