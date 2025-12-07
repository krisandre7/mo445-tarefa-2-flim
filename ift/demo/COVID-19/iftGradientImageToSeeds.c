//
// Created by azaelmsousa on 06/10/20.
//

#include "ift.h"

int main(int argc, char *argv[])
{

    if (argc != 3){
        iftError("Usage: iftGradientToSeeds <gradient image> <output seeds>","main");
    }

    iftImage *grad = iftReadImageByExt(argv[1]);

    iftKernel *K = iftGaussianKernel(1.5,1);
    iftImage *smooth = iftLinearFilter(grad,K);

    int th = iftOtsu(grad);
    iftImage *bin = iftThreshold(smooth,1.2*th,iftMaximumValue(grad),1);
    iftSet *S = NULL;
    iftMaskImageToSet(bin,&S);
    iftDestroyImage(&grad);
    iftDestroyImage(&smooth);

    iftLabeledSet *L = NULL;
    while (S != NULL){
        int elem = iftRemoveSet(&S);
        iftInsertLabeledSet(&L,elem,0);
    }

    iftWriteSeeds(L,bin,argv[2]);
    iftDestroyLabeledSet(&L);
    iftDestroyImage(&bin);

    return 0;
}