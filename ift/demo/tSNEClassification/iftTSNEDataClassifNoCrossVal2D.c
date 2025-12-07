//
// Created by Cesar Castelo on Mar 22, 2018
//

#include <ift.h>

#define CLASS_SVM_LINEAR    0
#define CLASS_SVM_RBF       1
#define CLASS_OPF           2
#define CLASS_KNN           3

char *classMetName(int classMet, iftDict **classifParams)
{
    char *name = NULL, *svmMult = NULL;

    switch (iftGetLongValFromDict("svmMultType", classifParams[CLASS_SVM_LINEAR])) {
        case IFT_OVO:
            svmMult = iftCopyString("OVO"); break;
        case IFT_OVA:
            svmMult = iftCopyString("OVA"); break;
    }

    char k[10];
    sprintf(k, "%i", (int)iftGetLongValFromDict("k", classifParams[CLASS_KNN]));

    switch (classMet) {
        case CLASS_SVM_LINEAR:
            name = iftConcatStrings(2, "SVM-Linear-", svmMult); break;
        case CLASS_SVM_RBF:
            name = iftConcatStrings(2, "SVM-RBF-", svmMult); break;
        case CLASS_OPF:
            name = iftCopyString("OPF"); break;
        case CLASS_KNN:
            name = iftConcatStrings(2, "kNN-", k); break;
    }

    return name;
}

float *classification(iftDataSet *Z, int classMet, iftDict **classifParams)
{
    timer *t1 = iftTic();
    fprintf(stdout, "\n--> training/classification with %s (nTrainSamp: %d, nTestSamp: %d, nFeats: %d) ...\n", classMetName(classMet, classifParams), Z->ntrainsamples, (Z->nsamples - Z->ntrainsamples), Z->nfeats); fflush(stdout);
    if(classMet == CLASS_SVM_LINEAR || classMet == CLASS_SVM_RBF) {
        int svmType = IFT_NIL, svmMultType = IFT_NIL;
        if(classMet == CLASS_SVM_LINEAR) { svmType = IFT_LINEAR; svmMultType = iftGetLongValFromDict("svmMultType", classifParams[CLASS_SVM_LINEAR]); }
        if(classMet == CLASS_SVM_RBF) { svmType = IFT_RBF; svmMultType = iftGetLongValFromDict("svmMultType", classifParams[CLASS_SVM_RBF]); }
        iftSVM *svm = iftCreateSVM(svmType, svmMultType, 1e5, 1.0/(float)Z->nfeats);
        iftSVMTrain(svm, Z);
        iftSVMClassify(svm, Z, IFT_TEST);
        iftDestroySVM(svm); ////
    }
    else if(classMet == CLASS_OPF) {
        iftCplGraph *graph = iftCreateCplGraph(Z);
        iftSupTrain2(graph);
        iftClassify(graph, Z);
        iftDestroyCplGraph(&graph);
    }
    else if(classMet == CLASS_KNN) {

        iftDataSet *Z1 = iftExtractSamples(Z,IFT_TRAIN);
        int k = iftGetLongValFromDict("k", classifParams[CLASS_KNN]);
        iftClassifyByKNN(Z1, Z, k);
        iftDestroyDataSet(&Z1);
    }

    float *results = iftAllocFloatArray(2);
    //results[0] = iftTruePositives(Z);
    results[0] = iftCohenKappaScore(Z);
    results[1] = iftCompTime(t1, iftToc());
    fprintf(stdout, "Kappa score (on testing set): %f\n", results[0]); fflush(stdout);
    fprintf(stdout, "Time to perform classification: %s\n", iftFormattedTime(results[1])); fflush(stdout);

    return results;
}

iftDataSet *tsneProjection(iftDataSet *Z, int dim)
{
    timer *t1 = iftTic();
    fprintf(stdout, "--> Creating projection with t-SNE for dim: %d ...\n", dim); fflush(stdout);
    iftDataSet *Zproj = iftDimReductionByTSNE(Z, dim, 40, 1000, false);
    fprintf(stdout, "Time to perform t-SNE projection: %s\n", iftFormattedTime(iftCompTime(t1, iftToc()))); fflush(stdout);

    return Zproj;
}

int main(int argc, char** argv)  {

    if(argc != 3) {
        iftError("\nUsage: iftTSNEDataClassifNoCrossVal2D <...>\n"
                         "[1] dataset: dataset to be classified\n"
                         "[2] Percentage to be used for training [0..1]\n",
                 "iftTSNEDataClassifNoCrossVal2D.c");
    }

    /* read parameters */
    char *dataSetName = iftCopyString(argv[1]);
    float percTrain = atof(argv[2]);

    if(percTrain >= 1) {
        iftError("percTrain must be less than 1, since the testing set is used to perform the classification task", "iftTSNEDataClassif.c");
    }

    /* read the dataset and split it into training and testing sets */
    fprintf(stdout, "============================================================================================================================\n"); fflush(stdout);
    fprintf(stdout, "--> Reading the dataset ...\n"); fflush(stdout);
    iftDataSet *Z  = iftReadDataSet(dataSetName);
    iftNormalizeSamples(Z);

    fprintf(stdout, "Num. Samples: %d\n", Z->nsamples); fflush(stdout);
    fprintf(stdout, "Num. Features: %d\n", Z->nfeats); fflush(stdout);

    iftIntArray *labels = iftGetDataSetTrueLabels(Z);
    iftSampler *trainSampler = iftStratifiedRandomSubsampling(labels->val, labels->n, 1, (float)(labels->n)*percTrain);
    iftDataSetSampling(Z, trainSampler, 0);
    fprintf(stdout, "Num. Training Samples: %d\n", Z->ntrainsamples); fflush(stdout);
    fprintf(stdout, "Num. Testing Samples: %d\n", (Z->nsamples - Z->ntrainsamples)); fflush(stdout);

    /* save results in a CSV file */
    int numbDim = 1, *dimArray = iftAllocIntArray(numbDim); dimArray[0] = 2;
    /*
    int numbDim = log(Z->nfeats), *dimArray = iftAllocIntArray(numbDim), dim = (int)((float)Z->nfeats/2.0);
    dim = 2;
    for(int i = 0; i < numbDim; i++) {
        dimArray[i] = dim;
        //dim = (int)((float)dim/2.0);
        dim *= 2;
    }
     */

    int numbClassif = 4, csvCount = 0;
    iftCSV *csvFile = iftCreateCSV((numbDim+1)*numbClassif+1, 4);
    sprintf(csvFile->data[csvCount][0], "dim");
    sprintf(csvFile->data[csvCount][1], "method");
    sprintf(csvFile->data[csvCount][2], "classif_kappa");
    sprintf(csvFile->data[csvCount][3], "classif_time");
    csvCount++;

    /* parameters for the classifiers */
    iftDict **classifParams = iftAlloc(numbClassif, sizeof(iftDict*));
    for(int classMet = 0; classMet < numbClassif; classMet++)
        classifParams[classMet] = iftCreateDict();

    iftInsertIntoDict("svmMultType", IFT_OVO, classifParams[CLASS_SVM_LINEAR]);
    iftInsertIntoDict("svmMultType", IFT_OVO, classifParams[CLASS_SVM_RBF]);
    iftInsertIntoDict("k", 1, classifParams[CLASS_KNN]);

    /* perform training/classification on the original dimension */
    for(int classMet = 0; classMet < numbClassif; classMet++) {
        float *origDimRes = classification(Z, classMet, classifParams);

        sprintf(csvFile->data[csvCount][0], "%d", Z->nfeats);
        sprintf(csvFile->data[csvCount][1], "%s", classMetName(classMet, classifParams));
        sprintf(csvFile->data[csvCount][2], "%f", origDimRes[0]);
        sprintf(csvFile->data[csvCount][3], "%f", origDimRes[1]/1000.0);

        csvCount++;
        iftFree(origDimRes);
    }

    /* perform t-SNE projections into several dimensions */
    int *bestDim = iftAllocIntArray(numbClassif);
    char fileName[512];
    float *bestDimKappa = iftAllocFloatArray(numbClassif);
    for(int d = 0; d < numbDim; d++) {
        fprintf(stdout, "\n----------------------------------------------------------------------------------------------------------------------------\n"); fflush(stdout);
        int dim = dimArray[d];

        /* perform t-SNE projection */
        iftDataSet *Zproj = tsneProjection(Z, dim);

        /* perform training/classification in the t-SNE dimension */
        for(int classMet = 0; classMet < numbClassif; classMet++) {
            float *tsneDimRes = classification(Zproj, classMet, classifParams);

            sprintf(csvFile->data[csvCount][0], "%d", dim);
            sprintf(csvFile->data[csvCount][1], "%s", classMetName(classMet, classifParams));
            sprintf(csvFile->data[csvCount][2], "%f", tsneDimRes[0]);
            sprintf(csvFile->data[csvCount][3], "%f", tsneDimRes[1]/1000.0);

            if(tsneDimRes[0] > bestDimKappa[classMet]) {
                bestDimKappa[classMet] = tsneDimRes[0];
                bestDim[classMet] = dim;
            }

            csvCount++;
            iftFree(tsneDimRes);
        }

        csvCount++;
        iftDestroyDataSet(&Zproj);
    }
    fprintf(stdout, "\n----------------------------------------------------------------------------------------------------------------------------\n"); fflush(stdout);
    for(int classMet = 0; classMet < numbClassif; classMet++)
        fprintf(stdout, "Best kappa score (on testing set) for %s: %f, found with dimension: %d\n", classMetName(classMet, classifParams), bestDimKappa[classMet], bestDim[classMet]); fflush(stdout);

    sprintf(fileName, "t-sne-classif-results_%s.csv", iftFilename(dataSetName, NULL));
    iftWriteCSV(csvFile, fileName, ';');

    iftDestroyDataSet(&Z);
    iftDestroyIntArray(&labels);
    iftDestroySampler(&trainSampler);
    iftDestroyCSV(&csvFile);

    return 0;
}