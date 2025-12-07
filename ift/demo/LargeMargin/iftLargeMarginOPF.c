#include "ift.h"

iftDataSet *iftLargeMarginDataSet(iftDataSet *Z, int k_targets, double learning_rate, int max_iter,
                                  float accept_prob, bool verbose);

iftDataSet *iftKernelLMDataSet(iftDataSet *Z, int d_out, int k_targets, double learning_rate, int max_iter, double c,
                               iftKernelFunction *K, double param1, double param2, float accept_prob, bool verbose);

int main(int argc, const char *argv[])
{
    if (argc != 5) {
        iftError("iftLargeMarginOPF <img path> <seeds path> <out path> <learning rate>", "iftLargeMarginOPF");
    }

    const char *img_path = argv[1];
    const char *seeds_path = argv[2];
    const char *out_path = argv[3];
    double learn_rate = atof(argv[4]);

    iftImage *img = iftReadImageByExt(img_path);
    iftLabeledSet *seeds = iftReadSeeds(img, seeds_path);
    iftMImage *mimg = iftImageToMImage(img, LAB_CSPACE);

    iftDataSet *Z = iftMImageToDataSet(mimg, NULL, 0.25);
    iftLabelDataSetFromSeeds(Z, seeds, NULL);

    iftDataSet *Z_lm = iftLargeMarginDataSet(Z, 100, learn_rate, 1000, 0.1, true);
//    iftKernelFunction *K = iftMetricLearnKernel(KERNEL_GAUSSIAN);
//    iftDataSet *Z_lm = iftKernelLMDataSet(Z, 2, 10, learn_rate, 1000, 1, K, 1e3, 0, 0.1, true);
    iftDestroyDataSet(&Z);
    Z = Z_lm;

    iftDataSet *Z_train = iftExtractSamples(Z, IFT_SUPERVISED);

    iftSetStatus(Z_train, IFT_TRAIN);
    iftCplGraph *graph = iftCreateCplGraph(Z_train);
    iftSupTrain(graph);

    iftSetStatus(Z, IFT_TEST);
    iftClassifyWithCertaintyValues(graph, Z);

    iftImage *res = iftDataSetObjectMap(Z, NULL, 255, 2);

    iftWriteImageByExt(res, out_path);

    iftDestroyImage(&img);
    iftDestroyImage(&res);
    iftDestroyLabeledSet(&seeds);
    iftDestroyMImage(&mimg);
    iftDestroyDataSet(&Z);
    iftDestroyDataSet(&Z_train);
    iftDestroyCplGraph(&graph);

    return 0;
}


iftDataSet *iftLargeMarginDataSet(iftDataSet *Z, int k_targets, double learning_rate, int max_iter,
        float accept_prob, bool verbose)
{
    int n = iftCountSamples(Z, IFT_SUPERVISED);

    if (n == 0) {
        iftError("No supervised samples", "iftLargeMarginDataSet");
    }

    double *X = iftAlloc(n * Z->nfeats, sizeof *X);
    int *y = iftAlloc(n, sizeof *y);
    n = 0;
    for (int i = 0; i < Z->nsamples; i++)
    {
        if (iftHasSampleStatus(Z->sample[i], IFT_SUPERVISED))
        {
            double r_unif = ((double) random()) / RAND_MAX;
            if (r_unif <= accept_prob) {
                int row = n * Z->nfeats;
                for (int k = 0; k < Z->nfeats; k++) {
                    X[row + k] = Z->sample[n].feat[k];
                }
                y[n] = Z->sample[i].truelabel - 1;
                n++;
            }
        }
    }
    X = iftRealloc(X, n * Z->nfeats * (sizeof *X));

    iftCheckNumberOfTargets(y, n, &k_targets);

    double *L = iftLMCompAnalysis(X, y, NULL, n, Z->nfeats, Z->nfeats, k_targets, learning_rate, max_iter, verbose);

    iftFree(X);
    iftFree(y);

    iftDataSet *Z_out = iftCopyDataSet(Z, true);
    iftFree(Z_out->data->val);
    Z_out->data->val = iftSpaceTransformFloat(Z->data->val, L, Z->nsamples, Z->nfeats, Z->nfeats);
    for (int i = 0; i < Z_out->nsamples; i++) {
        Z_out->sample[i].feat = &Z_out->data->val[i * Z->nfeats];
    }

    iftFree(L);

    return Z_out;
}


iftDataSet *iftKernelLMDataSet(iftDataSet *Z, int d_out, int k_targets, double learning_rate, int max_iter, double c,
                               iftKernelFunction *K, double param1, double param2, float accept_prob, bool verbose)
{
    int n = iftCountSamples(Z, IFT_SUPERVISED);

    if (n == 0) {
        iftError("No supervised samples", "iftLargeMarginDataSet");
    }

    double *X = iftAlloc(n * Z->nfeats, sizeof *X);
    int *y = iftAlloc(n, sizeof *y);
    n = 0;
    for (int i = 0; i < Z->nsamples; i++)
    {
        if (iftHasSampleStatus(Z->sample[i], IFT_SUPERVISED))
        {
            double r_unif = ((double) random()) / RAND_MAX;
            if (r_unif <= accept_prob) {
                int row = n * Z->nfeats;
                for (int k = 0; k < Z->nfeats; k++) {
                    X[row + k] = Z->sample[n].feat[k];
                }
                y[n] = Z->sample[i].truelabel - 1;
                n++;
            }
        }
    }
    X = iftRealloc(X, n * Z->nfeats * (sizeof *X));

    iftCheckNumberOfTargets(y, n, &k_targets);

    if (verbose)
        printf("Computing gram matrix ...\t");
    double *gram = iftGramianMatrix(X, n, Z->nfeats, K, param1, param2);
    if (verbose) printf("done\n");

    double *omega = iftKernelLMCA(gram, y, n, d_out, k_targets, c, learning_rate, max_iter, verbose);

    iftFree(gram);
    iftFree(y);

    double *X_test = iftAlloc(Z->nsamples * Z->nfeats, sizeof *X_test);
    for (int i = 0; i < Z->nsamples; i++) {
        int row = i * Z->nfeats;
        for (int j = 0; j < Z->nfeats; j++) {
            X_test[row + j] = Z->sample[i].feat[j];
        }
    }

    gram = iftKernelFeatures(X, X_test, n, Z->nsamples, Z->nfeats, K, param1, param2);
    iftFree(X);
    iftFree(X_test);

    X_test = iftSpaceTransform(gram, omega, Z->nsamples, n, d_out);
    iftFree(omega);

    iftDataSet *Z_out = iftCopyDataSet(Z, false);
    iftDestroyMatrix(&Z_out->data);
    Z_out->nfeats = d_out;
    Z_out->data = iftCreateMatrix(Z_out->nfeats, Z_out->nsamples);
    for (int i = 0; i < Z_out->nsamples; i++) {
        int row = i * Z_out->nfeats;
        Z_out->sample[i].feat = &Z_out->data->val[row];
        for (int j = 0; j < Z_out->nfeats; j++) {
            Z_out->sample[i].feat[j] = X_test[row + j];
        }
    }
    iftFree(X_test);

    return Z_out;
}
