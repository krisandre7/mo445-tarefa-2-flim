#include "ift.h"




int main(int argc, const char *argv[]) {
    iftRandomSeed(time(NULL));

    if (argc != 3)
        iftError("iftPrintDataSet <dataset.data> <n_train>", "main");

    iftDataSet *Z = iftReadOPFDataSet(argv[1]);
    printf("nsamples: %d, nfeats: %d, nclasses: %d\n", Z->nsamples, Z->nfeats, Z->nclasses);

    iftIntArray *labels = iftGetDataSetTrueLabels(Z);

    iftSampler *sampler = iftBalancedRandomSubsampling(labels->val, labels->n, 1, Z->nsamples * atof(argv[2]));

    iftSampleDataSet(Z, sampler, 0);

    // puts("");
    // printf("Z->ntrainsamples: %d\n", Z->ntrainsamples);
    // for (int s = 0; s < Z->nsamples; s++)
    //     if (Z->sample[s].status == IFT_TRAIN)
    //         printf("Label %d - %d\n", Z->sample[s].truelabel, s);

    iftDestroyDataSet(&Z);
    iftDestroySampler(&sampler);
    iftDestroyIntArray(&labels);

    return 0;
}










