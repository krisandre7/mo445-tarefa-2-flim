
#include "ift.h"


int main(int argc, const char *argv[]) {
    if ((argc != 3) && (argc != 4))
        iftError("iftClassifySupervoxels <train.data> <test.data> <normalize_datasets>", "main");

    puts("- Reading Datasets");
    iftDataSet *Ztrain = iftReadOPFDataSet(argv[1]);
    iftSetStatus(Ztrain, IFT_TRAIN);
    iftDataSet *Ztest = iftReadOPFDataSet(argv[2]);

    printf("Ztrain: n_samples: %d, n_feats: %d, n_classes: %d\n", Ztrain->nsamples, Ztrain->nfeats, Ztrain->nclasses);
    printf("Ztest: n_samples: %d, n_feats: %d\n\n", Ztest->nsamples, Ztest->nfeats);

    if (argc == 4) {
        puts("- Normalizing Datasets");
        iftDataSet *Ztrain_norm = iftNormalizeDataSet(Ztrain);
        iftDestroyDataSet(&Ztrain);
        Ztrain = Ztrain_norm;

        iftDataSet *Ztest_norm = iftNormalizeTestDataSet(Ztrain, Ztest);
        iftDestroyDataSet(&Ztest);
        Ztest = Ztest_norm;
    }

    puts("- Training OPF");
    iftCplGraph *opf = iftCreateCplGraph(Ztrain);
    iftSupTrain(opf);
    
    puts("- Classifying Test");
    iftClassify(opf, Ztest);

    float acc = iftTruePositives(Ztest);
    printf("\nacc = %f\n", acc);

    iftImage *super_img = iftReadImageByExt("out/super.scn");
    puts("[1]");
    iftImage *data_img  = iftCreateImage(super_img->xsize, super_img->ysize, super_img->zsize);
    puts("[2]");

    for (int p = 0; p < super_img->n; p++) {
        int c            = super_img->val[p] - 1;     // ex: supervoxel's label 2 has index [1] 
        data_img->val[p] = Ztest->sample[c].label - 1;
    }
    puts("[3]");
    iftWriteImage(data_img, "out/data_img.scn");
    puts("[4]");


    iftDestroyDataSet(&Ztrain);
    iftDestroyDataSet(&Ztest);
    iftDestroyCplGraph(&opf);

    return 0;
}









