#include "ift.h"


int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("iftClassifyImageByOPF <image> <svm> <output_image>", "main");

    timer *t1 = iftTic();

    iftImage *img = iftReadImageByExt(argv[1]);
    iftSVM *svm = iftReadSVM(argv[2]);

    puts("- Image to Dataset");

    iftAdjRel *A = (iftIs3DImage(img)) ? iftSpheric(5.0) : iftCircular(5.0);
    iftDataSet *Ztest = iftImageToDataSetByAdjacencyFeats(img, NULL, A);
    Ztest->nclasses = 1;
    
    puts("- Classifying Image");
    iftSVMLinearClassifyOVA(svm, Ztest, IFT_TEST, NULL);
    // for (int s = 0; s < Ztest->nsamples; s++)
    //     printf("Ztest->sample[%d].label = %d\n", s, Ztest->sample[s].label);

    puts("- Writing Image");
    iftWriteImageByExt(iftDataSetToLabelImage(Ztest, NULL, IFT_LABEL, true), argv[3]);

    puts(iftFormattedTime(iftCompTime(t1, iftToc())));

    return 0;
}


