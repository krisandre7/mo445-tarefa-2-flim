#include "ift.h"

int main(int argc, char *argv[]){

    if(argc != 4 && argc != 5)
        iftError("iftBooleanMapDistance:  <input image (.png, .scn. ppm)>"
                         " <int k (number of iterations)> <output image (.png, .scn. ppm)> <OPTIONAL seeds.txt>", "main");

    timer *t1 = iftTic();

    iftImage *img = NULL;
    iftMImage *mimg = NULL;
    iftFImage *output = NULL;
    iftLabeledSet *seeds = NULL;

    img = iftReadImageByExt(argv[1]);

    if(iftIsColorImage(img)){
        mimg = iftImageToMImage(img, LAB_CSPACE);

        /* Shifting all values to positive so we can use GQueue */
        for(int b = 0; b < mimg->m; b++){
            int min_value = iftMMinimumValue(mimg, b);

            for(int p = 0; p < mimg->n; p++){
                mimg->val[p][b] -= min_value;
            }
        }

    } else {
        mimg = iftImageToMImage(img, GRAY_CSPACE);
    }

    if(argc == 5){
        seeds = iftReadSeeds2D(argv[4], img);
    }

    output = iftBooleanMapDistance(mimg, seeds, atoi(argv[2]));

//    iftAdjRel *A = iftCircular(1.0f);
//    iftImage *basins = iftImageBasins( iftFImageToImage(output, 255), A);

    iftWriteImageByExt( iftFImageToImage(output, 255), argv[3]);
//    iftWriteImageByExt(basins, argv[3]);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
    iftDestroyFImage(&output);
    iftDestroyLabeledSet(&seeds);

    return 0;
}
