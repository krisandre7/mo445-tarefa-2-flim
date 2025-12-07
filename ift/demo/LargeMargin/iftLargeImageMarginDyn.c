#include "ift.h"

int main(int argc, char *argv[])
{
    if (argc != 10) {
        iftExit("iftLargeMarginDynamic <img path> <seeds path> <height> <min tree size> <k targets> <k impostors> <learning rate> <ite. between l.m.> <out path>", "iftLargeMarginDynamic");
    }

    const char *img_path = argv[1];
    const char *seeds_path = argv[2];
    int height = atoi(argv[3]);
    int min_tree_size = atoi(argv[4]);
    int k_targets = atoi(argv[5]);
    int k_impostors = atoi(argv[6]);
    double learn_rate = atof(argv[7]);
    int iterations = atoi(argv[8]);
    const char *out_path = argv[9];

    iftImage *img = iftReadImageByExt(img_path);
    iftLabeledSet *seeds = iftReadSeeds(img, seeds_path);
    iftAdjRel *A = iftCircular(1.0f);

    iftMImage *mimg = iftImageToMImage(img, LABNorm_CSPACE);

    timer *tic = iftTic();
    iftImage *segm = iftLargeMarginDynTreeSegm(mimg, A, seeds, height, false, min_tree_size, k_targets, k_impostors,
                                               learn_rate, iterations);
//    iftImage *segm = iftLargeMarginDynForestSegm(mimg, A, seeds, height, false, min_tree_size, k_targets, k_impostors,
//                                               learn_rate, iterations);
    timer *toc = iftToc();
    char *time = iftFormattedTime(iftCompTime(tic, toc));
    printf("%s\n", time);
    iftFree(time);

    iftImage *out = iftMask(img, segm);
    iftWriteImageByExt(out, out_path);
    iftDestroyImage(&out);
    iftDestroyImage(&segm);

    segm = iftDynamicSetMinRootPolicy(mimg, A, seeds, height, false);
    out = iftMask(img, segm);
    iftWriteImageByExt(out, "original.png");
    iftDestroyImage(&out);
    iftDestroyImage(&segm);

    iftDestroyAdjRel(&A);
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
    iftDestroyLabeledSet(&seeds);

    return 0;
}





