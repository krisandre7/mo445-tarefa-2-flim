#include "ift.h"


int main(int argc, const char *argv[]) {
    if ((argc != 3) && (argc != 4))
        iftError("iftMedianFilter <input_image> <output_image> <(optional) Adjacency Radius: Default: 1.0>", "main");

    iftImage *img = iftReadImageByExt(argv[1]);

    /* iftAdjRel *A1, *A2, *A3; */
    /* A1 = iftCircular(3.0); */
    /* A2 = iftCircular(5.0); */
    /* A3 = iftCircular(1.5); */
    /* iftImage *Red = iftImageRed(img); */
    /* iftImage *Rf  = iftMedianFilter(Red,A1);     */
    /* iftWriteImageByExt(Rf,"R.png"); */
    /* iftImage *Green = iftImageGreen(img); */
    /* iftImage *Gf  = iftMedianFilter(Green,A2);     */
    /* iftWriteImageByExt(Gf,"G.png"); */
    /* iftImage *Blue = iftImageBlue(img); */
    /* iftImage *Bf  = iftMedianFilter(Blue,A3);     */
    /* iftWriteImageByExt(Bf,"B.png"); */
    /* iftImage *final = iftCreateColorImage(img->xsize,img->ysize,img->zsize,iftImageDepth(img)); */
    /* for (int p=0; p < final->n; p++){ */
    /*   iftColor RGB, YCbCr; */
    /*   RGB.val[0] = Rf->val[p];  */
    /*   RGB.val[1] = Gf->val[p]; */
    /*   RGB.val[2] = Bf->val[p]; */
    /*   YCbCr      = iftRGBtoYCbCr(RGB,255); */
    /*   final->val[p] = YCbCr.val[0]; */
    /*   final->Cb[p] = YCbCr.val[1]; */
    /*   final->Cr[p] = YCbCr.val[2]; */
    /* } */
    /* iftWriteImageByExt(final, argv[2]); */
    /* return(0); */
    float adj_radius = (argc == 4) ? atof(argv[3]) : 1.0;
    iftAdjRel *A = (iftIs3DImage(img)) ? iftSpheric(adj_radius) : iftCircular(adj_radius);

    printf("- Filtering image %s by Median: adj. radius: %f\n", argv[1], adj_radius);
    iftImage *filt_img = iftMedianFilter(img, A);
    iftWriteImageByExt(filt_img, argv[2]);


    iftDestroyAdjRel(&A);
    iftDestroyImage(&img);

    return 0;
}






