//
// Created by azaelmsousa on 13/11/20.
//

#include "ift.h"

int main(int argc, char *argv[]) {

    if (argc != 5) {
        iftError("iftClassifySupervoxelByProbAtlas <...>\n"\
                 "[1] Input test mimage\n"\
                 "[2] Input test label image\n"\
                 "[3] Supervoxel atlas model (.zip)\n"\
                 "[4] Output image\n", "main");
    }

    iftMImage *mimg = iftReadMImage(argv[1]);
    iftImage *label = iftReadImageByExt(argv[2]);
    iftSupervoxelAtlas *atlas = iftReadSupervoxelAtlas(argv[3]);

    printf("max val: %lf\nmin val: %lf\n",iftMMaximumValue(mimg,0),iftMMinimumValue(mimg,0));

    iftImage *prob_map = iftProbMapBySupervoxelAtlas(mimg,label,atlas);
    iftDestroyMImage(&mimg);
    iftDestroySupervoxelAtlas(&atlas);

    iftWriteImageByExt(prob_map,argv[4]);
    iftDestroyImage(&prob_map);

    return 0;
}