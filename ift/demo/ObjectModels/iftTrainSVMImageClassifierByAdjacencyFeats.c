#include "ift.h"

iftIntArray *iftGetBackgroundSpelIndices(  iftImage *template_brain) {
    long n_bg_voxels = template_brain->n - iftCountObjectSpels(template_brain, 1);

    iftIntArray *bg_img_idxs = iftCreateIntArray(n_bg_voxels);
    long i = 0;
    for (int p = 0; p < template_brain->n; p++)
        if (template_brain->val[p] == 0)
            bg_img_idxs->val[i++] = p;

    return bg_img_idxs;
}

iftIntArray *iftGetObjectSpelIndices(  iftImage *template_brain) {
    long n_object_voxels = iftCountObjectSpels(template_brain, 1);

    iftIntArray *obj_img_idxs = iftCreateIntArray(n_object_voxels);
    long i = 0;
    for (int p = 0; p < template_brain->n; p++)
        if (template_brain->val[p] != 0)
            obj_img_idxs->val[i++] = p;

    return obj_img_idxs;
}




int main(int argc, const char *argv[]) {
    if (argc != 4)
        iftError("iftTrainOPFBrain <template> <markers_image> <out_svm.zip>", "main");

    iftRandomSeed(time(NULL));

    iftImage *template_img = iftReadImageByExt(argv[1]);
    iftImage *markers_img = iftReadImageByExt(argv[2]);
    
    iftAdjRel *A = (iftIs3DImage(template_img)) ? iftSpheric(5.0) : iftCircular(5.0);
    iftDataSet *Ztrain = iftImageToDataSetByAdjacencyFeats(template_img, markers_img, A);
    iftSetStatus(Ztrain, IFT_TRAIN);

    printf("Ztrain->nsamples: %d\n", Ztrain->nsamples);
    printf("Ztrain->ntrainsamples: %d\n", Ztrain->ntrainsamples);
    printf("Ztrain->nfeats: %d\n", Ztrain->nfeats);

    iftSVM *svm = iftCreateSVM(IFT_LINEAR, IFT_OVA, 1e-5, 1.0 / Ztrain->nfeats); // create SVM
    puts("Training SVM");
    iftSVMTrainOVA(svm, Ztrain, Ztrain);
    puts("Writing SVM");
    iftWriteSVM(svm, argv[3]);

    iftDestroyAdjRel(&A);

    return 0;
}









