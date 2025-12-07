//
// Created by Cesar Castelo on Mar 13, 2018
//

#include <ift.h>

#define CLASS_SVM_LINEAR    0
#define CLASS_SVM_RBF       1
#define CLASS_OPF           2

int main(int argc, char** argv)  {

    if(argc != 5) {
        iftError("\nUsage: iftTSNEDataClassif <...>\n"
                         "\n[1] dataset: dataset to be classified\n"
                         "\n[2] classification method\n"
                         "    0: SVM (linear) [CLASS_SVM_LINEAR]\n"
                         "    1: SVM (RBF) [CLASS_SVM_RBF]\n"
                         "    2: OPF (supervised) [CLASS_OPF]\n"
                         "\n[3] Percentage to be used for training [0..1]\n"
                         "\n[4] Number of subsets to split the testing dataset\n",
                 "iftTSNEDataClassif.c");
    }

    /* read parameters */
    int classMet = atoi(argv[2]);
    float percTrain = atof(argv[3]);
    int numbTestSubsets = atoi(argv[4]);

    if(percTrain >= 1) {
        iftError("percTrain must be less than 1, since the testing set is used to perform the classification task", "iftTSNEDataClassif.c");
    }

    if(numbTestSubsets < 2) {
        iftError("The number of testing subsets must be at least 2", "iftTSNEDataClassif.c");
    }

    /* read the dataset and split it into training and testing sets */
    fprintf(stdout, "============================================================================================================================\n"); fflush(stdout);
    fprintf(stdout, "--> Reading the dataset ...\n"); fflush(stdout);
    iftDataSet *Z  = iftReadDataSet(argv[1]);
    iftNormalizeSamples(Z);

    fprintf(stdout, "Num. Samples: %d\n", Z->nsamples); fflush(stdout);
    fprintf(stdout, "Num. Features: %d\n", Z->nfeats); fflush(stdout);

    iftIntArray *labels = iftGetDataSetTrueLabels(Z);
    iftSampler *trainSampler = iftStratifiedRandomSubsampling(labels->val, labels->n, 1, (float)(labels->n)*percTrain);
    iftDataSetSampling(Z, trainSampler, 0);
    iftDataSet *Ztrain = iftExtractSamples(Z, IFT_TRAIN);
    iftDataSet *Ztest = iftExtractSamples(Z, IFT_TEST);
    fprintf(stdout, "Num. Samples used for learning: %d\n", Ztrain->nsamples); fflush(stdout);
    fprintf(stdout, "Num. Features: %d\n", Ztrain->nfeats); fflush(stdout);

    /* save results in a CSV file */
    iftCSV *csvFile = iftCreateCSV(5, ceil(log2(Z->nfeats)) + 1);
    sprintf(csvFile->data[0][0], "dim");
    sprintf(csvFile->data[0][1], "cross_val_acc");
    sprintf(csvFile->data[0][2], "cross_val_kappa");
    sprintf(csvFile->data[0][3], "cross_val_true_pos");
    sprintf(csvFile->data[0][4], "cross_val_time");

    /* perform t-SNE projections into several dimensions */
    int csvCount = 1, bestDim = IFT_NIL;
    char fileName[512];
    float bestDimAcc = IFT_INFINITY_FLT_NEG;
    for(int dim = Z->nfeats; dim >= 2; dim /= 2) {
        fprintf(stdout, "\n----------------------------------------------------------------------------------------------------------------------------\n"); fflush(stdout);
        iftDataSet *ZtrainProj = NULL;

        timer *t1 = iftTic();
        if(dim == Z->nfeats)
            ZtrainProj = iftCopyDataSet(Ztrain, true);
        else {
            fprintf(stdout, "--> Creating the projection for the training/validation set with t-SNE for dim=%d ...\n", dim); fflush(stdout);
            ZtrainProj = iftDataSetFromTSNEProjection(Ztrain, dim);
        }
        double tsneTime = iftCompTime(t1, iftToc());
        fprintf(stdout, "Time to perform t-SNE projection: %s\n", iftFormattedTime(tsneTime)); fflush(stdout);
        sprintf(csvFile->data[csvCount][0], "%d", dim);

        /* perform k-fold cross-validation with the chosen classifier */
        iftSampler *crossValidSampler = iftKFold(ZtrainProj->nsamples, 5);
        float accArray[crossValidSampler->niters];
        float kappaArray[crossValidSampler->niters];
        float truePosArray[crossValidSampler->niters];
        timer *t2 = iftTic();

        if(classMet == CLASS_SVM_LINEAR || classMet == CLASS_SVM_RBF) {
            int svmType = IFT_NIL;
            if(classMet == CLASS_SVM_LINEAR) {
                fprintf(stdout, "\n--> k-fold cross-validation with SVM-Linear (nSamples=%d, nFeats=%d) ...\n", ZtrainProj->nsamples, ZtrainProj->nfeats); fflush(stdout);
                svmType = IFT_LINEAR;
            }
            if(classMet == CLASS_SVM_RBF) {
                fprintf(stdout, "\n--> k-fold cross-validation with SVM-RBF (nSamples=%d, nFeats=%d) ...\n", ZtrainProj->nsamples, ZtrainProj->nfeats); fflush(stdout);
                svmType = IFT_RBF;
            }
#pragma omp parallel for
            for(int it = 0; it < crossValidSampler->niters; ++it) {
                iftSVM *svm = iftCreateSVM(svmType, IFT_OVA, 1e5, 1.0/(float)ZtrainProj->nfeats);
                iftDataSet *ZtrainProjCopy = iftCopyDataSet(ZtrainProj, true);
                iftDataSetSampling(ZtrainProjCopy, crossValidSampler, it);
                iftSVMTrain(svm, ZtrainProjCopy);
                iftSVMClassify(svm, ZtrainProjCopy, IFT_TEST);
                accArray[it] = iftClassifAccuracy(ZtrainProjCopy);
                kappaArray[it] = iftCohenKappaScore(ZtrainProjCopy);
                truePosArray[it] = iftTruePositives(ZtrainProjCopy);
                iftDestroySVM(svm); ////
                iftDestroyDataSet(&ZtrainProjCopy);
            }
        }
        else if(classMet == CLASS_OPF) {
            fprintf(stdout, "\n--> k-fold cross-validation with Supervised OPF (nSamples=%d, nFeats=%d) ...\n", ZtrainProj->nsamples, ZtrainProj->nfeats); fflush(stdout);
#pragma omp parallel for
            for(int it = 0; it < crossValidSampler->niters; ++it) {
                iftDataSet *ZtrainProjCopy = iftCopyDataSet(ZtrainProj, true);
                iftDataSetSampling(ZtrainProjCopy, crossValidSampler, it);
                iftCplGraph *graph = iftCreateCplGraph(ZtrainProjCopy);
                iftSupTrain2(graph);
                iftClassify(graph, ZtrainProjCopy);
                accArray[it] = iftClassifAccuracy(ZtrainProjCopy);
                kappaArray[it] = iftCohenKappaScore(ZtrainProjCopy);
                truePosArray[it] = iftTruePositives(ZtrainProjCopy);
                iftDestroyCplGraph(&graph);
                iftDestroyDataSet(&ZtrainProjCopy);
            }
        }
        double crossValAcc = iftMeanFloatArray(accArray, crossValidSampler->niters);
        double crossValKappa = iftMeanFloatArray(kappaArray, crossValidSampler->niters);
        double crossValTruePos = iftMeanFloatArray(truePosArray, crossValidSampler->niters);
        double crossValTime = iftCompTime(t2, iftToc());
        fprintf(stdout, "Accuracy (on validation set): %f\n", crossValAcc); fflush(stdout);
        fprintf(stdout, "Cohen Kappa (on validation set): %f\n", crossValKappa); fflush(stdout);
        fprintf(stdout, "True positives (on validation set): %f\n", crossValTruePos); fflush(stdout);
        fprintf(stdout, "Time to perform k-fold cross-validation: %s\n", iftFormattedTime(crossValTime)); fflush(stdout);
        sprintf(csvFile->data[csvCount][1], "%f", crossValAcc);
        sprintf(csvFile->data[csvCount][2], "%f", crossValKappa);
        sprintf(csvFile->data[csvCount][3], "%f", crossValTruePos);
        sprintf(csvFile->data[csvCount][4], "%f", crossValTime);

        if(crossValTruePos > bestDimAcc) {
            bestDimAcc = crossValTruePos;
            bestDim = dim;
        }
        csvCount++;

        iftDestroyDataSet(&ZtrainProj);
        iftDestroySampler(&crossValidSampler);
    }
    fprintf(stdout, "\n----------------------------------------------------------------------------------------------------------------------------\n"); fflush(stdout);
    fprintf(stdout, "Best accuracy (on validation set): %f, found with dimension: %d\n", bestDimAcc, bestDim); fflush(stdout);

    /* split the testing set into several subsets (for speeding-up purposes) */
    iftDataSet **ZtestSubsets = NULL;
    numbTestSubsets = iftSplitDatasetIntoRandomlyDataSetArray(Ztest, numbTestSubsets, &ZtestSubsets, true);

    /* classify all the testing subsets using the best dimension (found in cross validation) */
    timer *t3 = iftTic();
    float accArray[numbTestSubsets];
    float kappaArray[numbTestSubsets];
    float truePosArray[numbTestSubsets];
    for(int t = 0; t < numbTestSubsets; t++) {
        fprintf(stdout, "\n----------------------------------------------------------------------------------------------------------------------------\n"); fflush(stdout);
        /* compute the reduction with t-SNE */
        iftDataSet *Zjoin = iftMergeDataSets(Ztrain, ZtestSubsets[t]);
        iftDataSet *ZjoinProj = NULL;

        if(bestDim != Z->nfeats) {
            fprintf(stdout, "\n--> Creating the projection for the testing subset %d with t-SNE for dim=%d ...\n", t, bestDim); fflush(stdout);
            ZjoinProj = iftDataSetFromTSNEProjection(Zjoin, bestDim);
        }
        else {
            fprintf(stdout, "\n--> The best accuracy was found with the original dimension ... using the original dimension\n"); fflush(stdout);
            ZjoinProj = iftCopyDataSet(Zjoin, true);
        }

        /* perform the classification */
        if(classMet == CLASS_SVM_LINEAR || classMet == CLASS_SVM_RBF) {
            int svmType = IFT_NIL;
            if(classMet == CLASS_SVM_LINEAR) {
                fprintf(stdout, "\n--> Classifying the testing subset %d with SVM-Linear for dim=%d (nSamples=%d) ...\n", t, bestDim, ZtestSubsets[t]->nsamples); fflush(stdout);
                svmType = IFT_LINEAR;
            }
            if(classMet == CLASS_SVM_RBF) {
                fprintf(stdout, "\n--> Classifying the testing subset %d with SVM-RBF for dim=%d (nSamples=%d) ...\n", t, bestDim, ZtestSubsets[t]->nsamples); fflush(stdout);
                svmType = IFT_RBF;
            }
            iftSVM* svm = iftCreateSVM(svmType, IFT_OVA, 1e5, 1.0/(float)ZjoinProj->nfeats);
            iftSVMTrain(svm, ZjoinProj);
            iftSVMClassify(svm, ZjoinProj, IFT_TEST);
            iftDestroySVM(svm); ////
        }
        else if(classMet == CLASS_OPF) {
            fprintf(stdout, "\n--> Classifying the testing subset %d with Supervised OPF for dim=%d (nSamples=%d) ...\n", t, bestDim, ZtestSubsets[t]->nsamples); fflush(stdout);
            iftCplGraph *graph = iftCreateCplGraph(ZjoinProj);
            iftSupTrain2(graph);
            iftClassify(graph, ZjoinProj);
            iftDestroyCplGraph(&graph);
        }
        accArray[t] = iftClassifAccuracy(ZjoinProj);
        kappaArray[t] = iftCohenKappaScore(ZjoinProj);
        truePosArray[t] = iftTruePositives(ZjoinProj);

        iftDestroyDataSet(&Zjoin);
        iftDestroyDataSet(&ZjoinProj);
    }

    fprintf(stdout, "\n----------------------------------------------------------------------------------------------------------------------------\n"); fflush(stdout);
    double testAcc = iftMeanFloatArray(accArray, numbTestSubsets);
    double testKappa = iftMeanFloatArray(kappaArray, numbTestSubsets);
    double testTruePos = iftMeanFloatArray(truePosArray, numbTestSubsets);
    double testTime = iftCompTime(t3, iftToc());
    fprintf(stdout, "Mean Accuracy (on testing set): %f\n", testAcc); fflush(stdout);
    fprintf(stdout, "Mean Cohen Kappa (on testing set): %f\n", testKappa); fflush(stdout);
    fprintf(stdout, "Mean True positives (on testing set): %f\n", testTruePos); fflush(stdout);
    fprintf(stdout, "Time to perform classification: %s\n", iftFormattedTime(testTime)); fflush(stdout);

    sprintf(fileName, "t-sne-classif-results_classif-%d.csv", classMet);
    iftWriteCSV(csvFile, fileName, ';');

    iftDestroyDataSet(&Z);
    iftDestroyIntArray(&labels);
    iftDestroySampler(&trainSampler);
    iftDestroyDataSet(&Ztrain);
    iftDestroyDataSet(&Ztest);
    iftDestroyCSV(&csvFile);

    return 0;
}