#include "ift.h"

//#define KERNEL_X 91.0
//#define KERNEL_Y 37.0

#define KERNEL_X 15.0
#define KERNEL_Y 21.0

float ativacao(float a, float b) {

    //background/foreground threshold
    a-=0.5;
    b-=0.5;

    float x = a*b;

    if(a<0 && b<0) {//the background was activating the output image too much
        x = -x;
    }

    //printf("%f -> %f\n", a, b);

    return x;
}

iftImage *iftBorderDetection2(iftImage *orig) {

    iftImage *aux[3];
    iftAdjRel *A = NULL;

    A = iftCircular(5.0);
    aux[0] = iftNormalizeImage(orig, A, 4095);

    aux[1] = iftCloseBasins(aux[0],NULL,NULL);
    aux[2] = iftSub(aux[1], aux[0]);

    iftDestroyImage(&aux[0]);
    iftDestroyImage(&aux[1]);
    iftDestroyAdjRel(&A);

    return aux[2];
}

iftFImage  *iftLinearFilter2DAtivacao(iftImage *img, iftKernel *K)
{
    iftImage *out;
    iftFImage *fimg = iftCreateFImage(img->xsize, img->ysize, img->zsize);
    iftAdjRel *A=K->A;

    float maxImg = iftMaximumValue(img);
    float maxK = iftMaxFloatArray(K->weight, K->A->n);

    for (int p=0; p < img->n; p++) {
        iftVoxel u   = iftGetVoxelCoord(img,p);
        double val = 0.0;
        for (int i=0; i < A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A,u,i);
            if (iftValidVoxel(img,v)){
                int q = iftGetVoxelIndex(img,v);
                //printf("%f\n", ativacao((float)img->val[q], K->weight[i]));
                val += ativacao((float)img->val[q]/maxImg, K->weight[i]/maxK);
            }
            else {
                //patches with a lot of pixels out of the kernel window should be penalized
                val -= 1;//image border
            }
        }
        //printf("%f\n", val);
        fimg->val[p] = val;
    }

    return(fimg);
}

int main(int argc, char *argv[])
{
    iftImage            *img, *img_ker, *qimg, *qimg_ker;
    iftFImage *img_filt;
    iftKernel           *ker;
    iftAdjRel           *A;
    int                 q;
    iftVoxel            u,v;
    //timer               *t1=NULL,*t2=NULL;

    /*--------------------------------------------------------*/
    /*
    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();
     */

    /*--------------------------------------------------------*/

    if (argc!=4){
        fprintf(stdout,"Usage: iftExtractFeatures <imagem> <imagem_kernel> <output>\n");
        exit(1);
    }

    img = iftReadImageByExt(argv[1]);
    img_ker = iftReadImageByExt(argv[2]);

    //qimg = iftQuantize(img, 4, 255);
    //qimg_ker = iftQuantize(img_ker, 4, 255);

    qimg = iftBorderDetection2(img);
    qimg_ker = iftBorderDetection2(img_ker);

    qimg_ker = iftInterp2D(qimg_ker, KERNEL_X/qimg_ker->xsize, KERNEL_Y/qimg_ker->ysize);//images must have odd size in order to have a proper center

    A = iftRectangular(qimg_ker->xsize, qimg_ker->ysize);
    printf("size A: %d, size img_ker: %d \n", A->n, img_ker->n);
    ker = iftCreateKernel(A);

    u.z = 0;
    u.x = qimg_ker->xsize/2;
    u.y = qimg_ker->ysize/2;

    for (int i=0; i < A->n; i++) {
        v = iftGetAdjacentVoxel(A, u, i);
        q = iftGetVoxelIndex(qimg_ker, v);
        ker->weight[i] = (float)qimg_ker->val[q];
    }


    float mean = iftMean(ker->weight, ker->A->n);
    float std = iftStddevFloatArray(ker->weight, ker->A->n);

//    for (int i = 0; i < ker->A->n; ++i) {
//        ker->weight[i] = (ker->weight[i]-mean)/std;
//    }
//
//    for (int i = 0; i < qimg->n; ++i) {
//        qimg->val[i] = (int)((qimg->val[i]-mean)/std);
//    }

    img_filt = iftLinearFilter2DAtivacao(qimg, ker);

    iftImage* iimg_filt = iftFImageToImage(img_filt, 255);
    iftWriteImageP5(iimg_filt, argv[3]);
    //iftWriteImageP5(iftNormalize(qimg_ker, 0, 255), "tempkernel.pgm");
    //iftWriteImageP5(iftNormalize(qimg, 0, 255), "tempimg.pgm");

    iftDestroyImage(&img);
    iftDestroyImage(&img_ker);
    iftDestroyImage(&qimg);
    iftDestroyImage(&qimg_ker);
    iftDestroyKernel(&ker);
    iftDestroyFImage(&img_filt);
    iftDestroyImage(&iimg_filt);
    iftDestroyAdjRel(&A);

    /* ---------------------------------------------------------- */

    /*
    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial,MemDinFinal);
               */

    return(0);
}




