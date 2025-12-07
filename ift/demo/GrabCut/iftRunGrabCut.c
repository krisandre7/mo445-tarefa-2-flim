#include "ift/segm/GrabCut.h"
#include "ift.h"


int main(int argc, char *argv[])
{
    if (argc != 5)
        iftExit("iftRunGrabCut <input img> <aux label img> <iterations> <out img>", "iftRunGrabCut");

    const char *imgpath = argv[1];
    const char *labpath = argv[2];
      int ite = atoi(argv[3]);
    const char *outpath = argv[4];

    iftImage *lab = iftReadImageByExt(labpath);
    iftImage *norm = iftNormalize(lab, 0, 1);
    iftImage *reg = iftMaybeForeground(norm);
    iftDestroyImage(&norm);
    iftDestroyImage(&lab);

    iftImage *img = iftReadImageByExt(imgpath);
    iftMImage *mimg = iftImageToMImage(img, RGBNorm_CSPACE);

//    iftMImage *mimg = iftImageToMImage(img, LABNorm_CSPACE);
//    for (int i = 0; i < mimg->n; i++) {
//        for (int j = 0; j < mimg->m; j++) {
//            mimg->val[i][j] /= 255;
//        }
//    }

    double beta = iftGraphCutBeta(mimg);
    printf("Beta: %lf\n", beta);

    timer *tic = iftTic();
    iftImage *segm = iftGrabCut(mimg, reg, beta, ite);
    timer *toc = iftToc();

    char *time = iftFormattedTime(iftCompTime(tic, toc));
    printf("%s\n", time);
    iftFree(time);

    iftImage *out = iftMask(img, segm);
    iftWriteImageByExt(out, outpath);

    iftDestroyImage(&reg);
    iftDestroyImage(&img);
    iftDestroyImage(&segm);
    iftDestroyImage(&out);
    iftDestroyMImage(&mimg);

    return 0;
}
