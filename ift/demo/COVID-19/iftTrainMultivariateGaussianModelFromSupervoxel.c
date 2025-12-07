//
// Created by azaelmsousa on 11/11/20.
//

#include "ift.h"

int main(int argc, char *argv[])
{

    if (argc != 4){
        iftError("iftTrainMultivariateGaussianModelFromSupervoxel <...>\n"\
                 "[1] Input mimage\n"\
                 "[2] Supervoxel label image\n"\
                 "[3] Output gaussian distribution\n","main");
    }

    iftMImage *mimg                = iftReadMImage(argv[1]);
    iftImage *svoxel               = iftReadImageByExt(argv[2]);

    puts("Creating Supervoxel Atlas");
    iftSupervoxelAtlas *A = iftTrainSupervoxelAtlas(mimg,svoxel);

    puts("Writing Atlas");
    iftWriteSupervoxelAtlas(A,argv[3]);

    puts("Destroying Atlas");
    iftDestroySupervoxelAtlas(&A);

    return 0;
}