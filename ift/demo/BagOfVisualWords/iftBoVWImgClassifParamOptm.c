//
// Created by Cesar Castelo on Feb 22, 2018
//

#include <ift.h>

#define OPTM_STRGY_RANDOM   0
#define OPTM_STRGY_GRID     1

void iftBovwCreateOptmParamDict(iftDict *optmParams, int intPointDetector, int localFeatExtractor, int dictEstimator, int codFunc)
{
    switch(intPointDetector) {
        case BOVW_INT_POINT_DETECTOR_RANDOM:
            iftInsertIntoDict("intPointDetec_nPoints", iftUniformIntSearchSpace(100, 200, 50), optmParams);
            //iftInsertIntoDict("intPointDetec_nPoints", iftUniformIntSearchSpace(50, 50, 50), optmParams);
            iftInsertIntoDict("intPointDetec_patchSize", iftUniformIntSearchSpace(10, 20, 5), optmParams);
            //iftInsertIntoDict("intPointDetec_patchSize", iftUniformIntSearchSpace(15, 15, 5), optmParams);
            break;
        case BOVW_INT_POINT_DETECTOR_DENSE:
            iftInsertIntoDict("intPointDetec_patchSize", iftUniformIntSearchSpace(10, 20, 5), optmParams);
            //iftInsertIntoDict("intPointDetec_patchSize", iftUniformIntSearchSpace(15, 15, 5), optmParams);
            break;
    }
    switch(localFeatExtractor) {
        case BOVW_LOCAL_FEAT_EXTRACTOR_RAW:
            iftInsertIntoDict("localFeatExtr_colSpaceOut", iftUniformIntSearchSpace(0, 2, 2), optmParams);
            //iftInsertIntoDict("localFeatExtr_colSpaceOut", iftUniformIntSearchSpace(0, 0, 2), optmParams);
            break;
        case BOVW_LOCAL_FEAT_EXTRACTOR_BIC:
            iftInsertIntoDict("localFeatExtr_nBins", iftUniformIntSearchSpace(4, 6, 2), optmParams);
            //iftInsertIntoDict("localFeatExtr_nBins", iftUniformIntSearchSpace(4, 4, 2), optmParams);
            break;
    }
    switch(dictEstimator) {
        case BOVW_DICT_ESTIMATOR_UNSUP_KMEANS:
        case BOVW_DICT_ESTIMATOR_SUP_PIX_KMEANS:
        case BOVW_DICT_ESTIMATOR_SUP_IMG_KMEANS:
        case BOVW_DICT_ESTIMATOR_SUP_PIX_AND_IMG_KMEANS:
            iftInsertIntoDict("dictEstim_nGroups", iftUniformIntSearchSpace(100, 300, 100), optmParams);
            //iftInsertIntoDict("dictEstim_nGroups", iftUniformIntSearchSpace(100, 100, 100), optmParams);
            break;
        case BOVW_DICT_ESTIMATOR_UNSUP_OPF:
        case BOVW_DICT_ESTIMATOR_SUP_PIX_OPF:
        case BOVW_DICT_ESTIMATOR_SUP_IMG_OPF:
        case BOVW_DICT_ESTIMATOR_SUP_PIX_AND_IMG_OPF:
            iftInsertIntoDict("dictEstim_knnGraphNeighbPerc", iftUniformDblSearchSpace(0.01, 0.10, 0.049), optmParams);
            //iftInsertIntoDict("dictEstim_knnGraphNeighbPerc", iftUniformDblSearchSpace(0.1, 0.1, 0.049), optmParams);
            iftInsertIntoDict("dictEstim_knownNumbGroups", iftUniformIntSearchSpace(0, 1, 1), optmParams);
            //iftInsertIntoDict("dictEstim_knownNumbGroups", iftUniformIntSearchSpace(0, 0, 1), optmParams);
            iftInsertIntoDict("dictEstim_nGroups", iftUniformIntSearchSpace(100, 300, 100), optmParams);
            //iftInsertIntoDict("dictEstim_nGroups", iftUniformIntSearchSpace(100, 100, 100), optmParams);
            break;
    }
    switch(codFunc) {
        case BOVW_COD_FUNC_SOFT_ASGMT:
            iftInsertIntoDict("codFunc_numbNearNeighb", iftUniformIntSearchSpace(5, 15, 5), optmParams);
            //iftInsertIntoDict("codFunc_numbNearNeighb", iftUniformIntSearchSpace(10, 10, 5), optmParams);
            iftInsertIntoDict("codFunc_wgtFunc", iftUniformIntSearchSpace(0, 1, 1), optmParams);
            //iftInsertIntoDict("codFunc_wgtFunc", iftUniformIntSearchSpace(0, 0, 1), optmParams);
            break;
    }
}


void iftBovwCreateDefaultParamDict(iftDict *defaultParams, int intPointDetector, int localFeatExtractor, int dictEstimator, int codFunc)
{
    switch(localFeatExtractor) {
        case BOVW_LOCAL_FEAT_EXTRACTOR_RAW:
            iftInsertIntoDict("localFeatExtr_colSpaceIn", YCbCr_CSPACE, defaultParams);
            break;
    }
    switch(dictEstimator) {
        case BOVW_DICT_ESTIMATOR_UNSUP_KMEANS:
        case BOVW_DICT_ESTIMATOR_SUP_PIX_KMEANS:
        case BOVW_DICT_ESTIMATOR_SUP_IMG_KMEANS:
        case BOVW_DICT_ESTIMATOR_SUP_PIX_AND_IMG_KMEANS:
            iftInsertIntoDict("dictEstim_retRealCent", 1, defaultParams);
            iftInsertIntoDict("dictEstim_clustSampPerc", 0.01, defaultParams);
            iftInsertIntoDict("dictEstim_maxIter", 20, defaultParams);
            iftInsertIntoDict("dictEstim_minImprov", 0.001, defaultParams);
            break;

        case BOVW_DICT_ESTIMATOR_UNSUP_OPF:
        case BOVW_DICT_ESTIMATOR_SUP_PIX_OPF:
        case BOVW_DICT_ESTIMATOR_SUP_IMG_OPF:
        case BOVW_DICT_ESTIMATOR_SUP_PIX_AND_IMG_OPF:
            iftInsertIntoDict("dictEstim_retRealCent", 1, defaultParams);
            iftInsertIntoDict("dictEstim_clustSampPerc", 0.01, defaultParams);
            break;
    }
    switch(codFunc) {
        case BOVW_COD_FUNC_HARD_ASGMT:
        case BOVW_COD_FUNC_SOFT_ASGMT:
            iftInsertIntoDict("codFunc_simFunc", BOVW_DIST_FUNC_1, defaultParams);
            iftInsertIntoDict("codFunc_normType", BOVW_NORM_TYPE_NONE, defaultParams);
            iftInsertIntoDict("codFunc_poolFunc", BOVW_POOL_FUNC_MAX, defaultParams);
            iftInsertIntoDict("codFunc_activFunc", BOVW_ACTIV_FUNC_ALL, defaultParams);
            break;
    }
}

double iftBovwEvalImgClassif(iftDict *bovwProblem, iftFileSet *fileSet, iftDict *optmParams)
{
    fprintf(stdout, "\n----------------------------------------------------------------------------------------------------------------------------\n"); fflush(stdout);
    fprintf(stdout, "optmParams:\n"); fflush(stdout);
    iftPrintDictAsArray(optmParams);

    /* create the BoVW structure and set the function pointers and their params */
    iftBagOfVisualWords *bovw = iftCreateBovw(iftGetStrValFromDict("pathToImgSet", bovwProblem));
    iftDict *params = iftMergeDicts(iftGetDictFromDict("defaultParams", bovwProblem), optmParams);
    iftBovwSetFunctionPointers(bovw, iftGetLongValFromDict("intPointDetector", bovwProblem), iftGetLongValFromDict("localFeatExtractor", bovwProblem),
                               iftGetLongValFromDict("dictEstimator", bovwProblem), iftGetLongValFromDict("codFunc", bovwProblem),
                               iftGetLongValFromDict("distFunc", bovwProblem), params);

    /* perform BoVW learning */
    fprintf(stdout, "--> Learning the Bag of Visual Words model ...\n"); fflush(stdout);
    iftBovwLearnImgClassif(bovw, (iftFileSet*)iftGetPtrValFromDict("trainImgFileSet", bovwProblem), iftGetStrValFromDict("labelImgSubDir", bovwProblem));
    fprintf(stdout, "Num. Features (Num. Visual Words): %d\n", bovw->dict->nrows); fflush(stdout);

    /* compute the final feature vectors */
    fprintf(stdout, "\n--> Computing the feature vectors ...\n"); fflush(stdout);
    iftDataSet *featVects = iftBovwComputeFeatVectsFromImgFileSet(bovw, (iftFileSet*)iftGetPtrValFromDict("trainImgFileSet", bovwProblem));
    iftNormalizeSamples(featVects);

    /* perform k-fold cross-validation with SVM */
    fprintf(stdout, "\n--> k-fold cross-validation with SVM ...\n"); fflush(stdout);
    iftSampler *crossValidSampler = (iftSampler*)iftGetPtrValFromDict("crossValidSampler", bovwProblem);
    float accArray[crossValidSampler->niters];
    timer *t1 = iftTic();
#pragma omp parallel for
    for(int it = 0; it < crossValidSampler->niters; ++it) {
        iftSVM* svm = iftCreateSVM(IFT_RBF, IFT_OVO, 1e5, 1.0/(float)featVects->nfeats);
        iftDataSet *featVectsCopy = iftCopyDataSet(featVects, true);
        iftDataSetSampling(featVectsCopy, crossValidSampler, it);
        //iftDataSet *featVectsCent = iftCentralizeDataSet(featVects);
        //iftDataSet *featVectsTrans = iftTransFeatSpaceBySupPCA(featVectsCent, 3);
        iftSVMTrain(svm, featVectsCopy);
        iftSVMClassify(svm, featVectsCopy, IFT_TEST);
        accArray[it] = iftTruePositives(featVectsCopy);

        //iftDestroyDataSet(&featVectsTrans);
        //iftDestroyDataSet(&featVectsCent);
        iftDestroySVM(svm); ////
        iftDestroyDataSet(&featVectsCopy);
    }
    double acc = iftMeanFloatArray(accArray, crossValidSampler->niters);
    fprintf(stdout, "Time to perform k-fold cross-validation: %s\n", iftFormattedTime(iftCompTime(t1, iftToc()))); fflush(stdout);
    fprintf(stdout, "Accuracy (on validation set): %f\n", acc); fflush(stdout);

    iftDestroyBovw(&bovw);
    iftDestroyDict(&params);
    iftDestroyDataSet(&featVects);

    return acc;
}

int main(int argc, char** argv)
{
    if(argc != 4) {
        iftError("\nUsage: iftBoVWImgClassifParamOptm <...>\n"
                         "[1] input_folder: Input folder with image set\n"
                         "[2] Percentage to be used for training [0..1]\n"
                         "[3] Optimization strategy\n"
                         "    0: Random search [OPTM_STRGY_RANDOM]\n"
                         "    1: Grid search [OPTM_STRGY_GRID]\n",
                 "iftBoVWImgClassifParamOptm.c");
    }

    /* read parameters */
    char *pathToImgSet = iftCopyString(argv[1]);
    float percTrain = atof(argv[2]);
    int optmStrgy = atoi(argv[3]);

    if(percTrain >= 1) {
        iftError("percTrain must be less than 1, since the testing set is used to perform the classification task", "iftBoVWImgClassifParamOptm.c");
    }

    /* read the image fileset */
    char origImgSubDir[] = "orig", labelImgSubDir[] = "label";
    fprintf(stdout, "============================================================================================================================\n"); fflush(stdout);
    fprintf(stdout, "--> Reading the image set ...\n"); fflush(stdout);
    fprintf(stdout, "============================================================================================================================\n"); fflush(stdout);
    char *imgDirPath = iftJoinPathnames(2, pathToImgSet, origImgSubDir);
    iftFileSet *imgFileSet = iftLoadFileSetFromDir(imgDirPath, 1);

    /* sample the training image fileset */
    int* labels = iftFileSetLabels(imgFileSet);
    iftSampler *trainSampler = iftStratifiedRandomSubsampling(labels, imgFileSet->n, 1, (float)(imgFileSet->n)*percTrain);
    iftSampleFileSet(imgFileSet, trainSampler, 0);
    iftFileSet *trainImgFileSet = iftExtractFileSamples(imgFileSet, IFT_TRAIN);
    iftFileSet *testImgFileSet = iftExtractFileSamples(imgFileSet, IFT_TEST);
    fprintf(stdout, "Num. Images used for learning: %d\n", (int)trainImgFileSet->n); fflush(stdout);

    /* sampler to perform k-fold cros-validation */
    iftSampler *crossValidSampler = iftKFold(trainImgFileSet->n, 5);

    /* CSV file to save the best parameters of each combination */
    int numbIntPointDetecMet = 2, numbLocalFeatExtrMet = 2, numbDictEstimMet = 8, numbCodFuncMet = 2;
    iftCSV *csvFile = iftCreateCSV((numbIntPointDetecMet*numbLocalFeatExtrMet*numbDictEstimMet*numbCodFuncMet) + 3, 9); // 3 = (field_names, best_param_combination_fieldnames, best_param_combination)
    sprintf(csvFile->data[0][0], "int_point_detector");
    sprintf(csvFile->data[0][1], "local_feat_extractor");
    sprintf(csvFile->data[0][2], "dict_estimator");
    sprintf(csvFile->data[0][3], "cod_func");
    sprintf(csvFile->data[0][4], "default_params");
    sprintf(csvFile->data[0][5], "best_params");
    sprintf(csvFile->data[0][6], "optm_cross_val_acc");
    sprintf(csvFile->data[0][7], "test_acc");
    sprintf(csvFile->data[0][8], "mean_eval_time");
    int csvCount = 1;

    /* optimize parameters for each combination of intPointDetector, localFeatExtractor, dictEstimator, codFunc */
    double maxTestAcc = IFT_INFINITY_DBL_NEG;
    int maxTestAccIdx = -1;
    int distFunc = BOVW_DIST_FUNC_1; // default value for distFunc
    for(int intPointDetector = 0; intPointDetector < numbIntPointDetecMet; intPointDetector++) {
        for(int localFeatExtractor = 1; localFeatExtractor < numbLocalFeatExtrMet; localFeatExtractor++) {
            for(int dictEstimator = 0; dictEstimator < numbDictEstimMet; dictEstimator++) {
                for(int codFunc = 0; codFunc < 1; codFunc++) {
                    if((dictEstimator != BOVW_DICT_ESTIMATOR_SUP_PIX_AND_IMG_KMEANS)
                       && (dictEstimator != BOVW_DICT_ESTIMATOR_SUP_PIX_AND_IMG_OPF)) {
                        /* create the optimization values for each parameter (according to each method) */
                        iftDict *optmParams = iftCreateDict();
                        iftBovwCreateOptmParamDict(optmParams, intPointDetector, localFeatExtractor, dictEstimator, codFunc);

                        /* create the default values for the parameters that will not be optimized */
                        iftDict *defaultParams = iftCreateDict();
                        iftBovwCreateDefaultParamDict(defaultParams, intPointDetector, localFeatExtractor, dictEstimator, codFunc);

                        /* create the BoVW Problem for the parameter optimization process */
                        iftDict *bovwProblem = iftCreateDict();
                        iftInsertIntoDict("pathToImgSet", pathToImgSet, bovwProblem);
                        iftInsertIntoDict("trainImgFileSet", trainImgFileSet, bovwProblem);
                        iftInsertIntoDict("labelImgSubDir", labelImgSubDir, bovwProblem);
                        iftInsertIntoDict("crossValidSampler", crossValidSampler, bovwProblem);
                        iftInsertIntoDict("intPointDetector", intPointDetector, bovwProblem);
                        iftInsertIntoDict("localFeatExtractor", localFeatExtractor, bovwProblem);
                        iftInsertIntoDict("dictEstimator", dictEstimator, bovwProblem);
                        iftInsertIntoDict("codFunc", codFunc, bovwProblem);
                        iftInsertIntoDict("distFunc", distFunc, bovwProblem);
                        iftInsertIntoDict("defaultParams", defaultParams, bovwProblem);

                        /* perform the parameter optimization process */
                        fprintf(stdout, "\n============================================================================================================================\n"); fflush(stdout);
                        fprintf(stdout, "--> Optimizing parameters ...\n"); fflush(stdout);
                        fprintf(stdout, "============================================================================================================================\n"); fflush(stdout);
                        iftBovwPrintChosenMethods(intPointDetector, localFeatExtractor, dictEstimator, codFunc);
                        iftDict *bestParams = NULL;

                        switch(optmStrgy) {
                            case OPTM_STRGY_RANDOM:
                                bestParams = iftRandomSearchDescriptor(optmParams, iftBovwEvalImgClassif, trainImgFileSet, 30, bovwProblem);
                                break;
                            case OPTM_STRGY_GRID:
                                bestParams = iftGridSearchDescriptor(optmParams, iftBovwEvalImgClassif, trainImgFileSet, bovwProblem);
                                break;
                        }

                        fprintf(stdout, "\n****************************************************************************************************************************\n"); fflush(stdout);
                        fprintf(stdout, "--> Optimum parameters for: \n"); fflush(stdout);
                        iftBovwPrintChosenMethods(intPointDetector, localFeatExtractor, dictEstimator, codFunc);
                        fprintf(stdout, "Parameters: \n"); fflush(stdout);
                        iftPrintDictAsArray(bestParams);

                        double optmCrossValAcc = iftGetDblValFromDict("best_func_val", bestParams);
                        fprintf(stdout, "Optimum accuracy (on validation set): %f\n", optmCrossValAcc); fflush(stdout);
                        double meanEvalTime = iftGetDblValFromDict("mean_eval_time", bestParams);
                        fprintf(stdout, "Mean training/validation time (BoVW training and SVM classification): %f\n", optmCrossValAcc); fflush(stdout);

                        /* create and train a BoVW structure with the optimum parameters and compute the feature vectors */
                        fprintf(stdout, "\n--> Training a BoVW model with the optimum parameters: \n"); fflush(stdout);
                        iftBagOfVisualWords *bovw = iftCreateBovw(pathToImgSet);
                        iftDict *params = iftMergeDicts(defaultParams, bestParams);
                        iftBovwSetFunctionPointers(bovw, intPointDetector, localFeatExtractor, dictEstimator, codFunc, distFunc, params);
                        iftBovwLearnImgClassif(bovw, trainImgFileSet, labelImgSubDir);
                        fprintf(stdout, "Num. Features (Num. Visual Words): %d\n", bovw->dict->nrows); fflush(stdout);

                        fprintf(stdout, "\n--> Computing the feature vectors for the training set: \n"); fflush(stdout);
                        iftDataSet *Ztrain = iftBovwComputeFeatVectsFromImgFileSet(bovw, trainImgFileSet);
                        iftSetStatus(Ztrain, IFT_TRAIN);

                        fprintf(stdout, "\n--> Computing the feature vectors for the testing set: \n"); fflush(stdout);
                        iftDataSet *Ztest = iftBovwComputeFeatVectsFromImgFileSet(bovw, testImgFileSet);
                        iftSetStatus(Ztest, IFT_TEST);

                        iftDataSet *Zfinal = iftMergeDataSets(Ztrain, Ztest);
                        iftNormalizeSamples(Zfinal);

                        /* perform SVM classification with the optimum parameters (using the testing set) */
                        fprintf(stdout, "\n--> Classifying the testing set with SVM: \n"); fflush(stdout);
                        iftSVM* svm = iftCreateSVM(IFT_RBF, IFT_OVO, 1e5, 1.0/(float)Zfinal->nfeats);
                        iftSVMTrain(svm, Zfinal);
                        iftSVMClassify(svm, Zfinal, IFT_TEST);
                        double testAcc = iftTruePositives(Zfinal);
                        fprintf(stdout, "Accuracy (on testing set): %f\n", testAcc); fflush(stdout);

                        if(testAcc > maxTestAcc) {
                            maxTestAcc = testAcc;
                            maxTestAccIdx = csvCount;
                        }

                        /* save the optimum parameters for the current combination in a CSV file */
                        iftRemoveValFromDict("best_func_val", bestParams);
                        sprintf(csvFile->data[csvCount][0], "%d", intPointDetector);
                        sprintf(csvFile->data[csvCount][1], "%d", localFeatExtractor);
                        sprintf(csvFile->data[csvCount][2], "%d", dictEstimator);
                        sprintf(csvFile->data[csvCount][3], "%d", codFunc);
                        sprintf(csvFile->data[csvCount][4], "%s", iftDictToStringMinified(defaultParams));
                        sprintf(csvFile->data[csvCount][5], "%s", iftDictToStringMinified(bestParams));
                        sprintf(csvFile->data[csvCount][6], "%f", optmCrossValAcc);
                        sprintf(csvFile->data[csvCount][7], "%f", testAcc);
                        sprintf(csvFile->data[csvCount][8], "%f", meanEvalTime);
                        csvCount++;

                        /* create a file with the training/evaluation accuracy */
                        char fileName[256];
                        sprintf(fileName, "accuracy_inte-%d_feat-%d_dict-%d_codi-%d_dist-%d.acc", intPointDetector, localFeatExtractor, dictEstimator, codFunc, distFunc);
                        FILE *filePtr = fopen(fileName, "w");
                        fprintf(filePtr, "optmCrossValAcc; testAcc\n");
                        fprintf(filePtr, "%f; %f\n", optmCrossValAcc, testAcc);
                        fclose(filePtr);

                        /* create a 2D projection with t-SNE with the feature vectors obtained */
                        fprintf(stdout, "\n--> Creating the projections with t-SNE ...\n"); fflush(stdout);

                        fprintf(stdout, "- t-SNE projection for the training/validation set\n"); fflush(stdout);
                        iftDataSet *Zfinal1 = iftRemoveClassFromDataSet(Zfinal, 7); // WARNING: remove the 7th class (only valid for the Protozoan image set)

                        iftDataSet *ZtrainFinal1 = iftExtractSamples(Zfinal1, IFT_TRAIN);
                        iftDataSet *ZtrainFinal2D = iftDimReductionByTSNE(ZtrainFinal1, 2, 40, 1000);
                        iftImage *trainImg2D = iftDraw2DFeatureSpace(ZtrainFinal2D, CLASS, (uchar)ALL);
                        sprintf(fileName, "projection_train_6-classes_inte-%d_feat-%d_dict-%d_codi-%d_dist-%d.png", intPointDetector, localFeatExtractor, dictEstimator, codFunc, distFunc);
                        iftWriteImageByExt(trainImg2D, fileName);
                        iftDestroyDataSet(&ZtrainFinal1);
                        iftDestroyDataSet(&ZtrainFinal2D);
                        iftDestroyImage(&trainImg2D);

                        /*
                        fprintf(stdout, "- t-SNE projection for the testing set\n"); fflush(stdout);
                        iftDataSet *ZtestFinal1 = iftExtractSamples(Zfinal1, IFT_TEST);
                        iftDataSet *ZtestFinal2D = iftDimReductionByTSNE(ZtestFinal1, 2, 40, 1000);
                        iftImage *testImg2D = iftDraw2DFeatureSpace(ZtestFinal2D, CLASS, (uchar)ALL);
                        sprintf(fileName, "projection_test_6-classes_inte-%d_feat-%d_dict-%d_codi-%d_dist-%d.png", intPointDetector, localFeatExtractor, dictEstimator, codFunc, distFunc);
                        iftWriteImageByExt(testImg2D, fileName);
                        iftDestroyDataSet(&ZtestFinal1);
                        iftDestroyDataSet(&ZtestFinal2D);
                        iftDestroyImage(&testImg2D);
                         */

                        iftDestroyDict(&optmParams);
                        iftDestroyDict(&defaultParams);
                        iftDestroyDict(&bovwProblem);
                        iftDestroyDict(&bestParams);
                        iftDestroyDict(&params);
                        iftDestroyBovw(&bovw);
                        iftDestroyDataSet(&Ztrain);
                        iftDestroyDataSet(&Ztest);
                        iftDestroyDataSet(&Zfinal);
                        iftDestroyDataSet(&Zfinal1);
                        iftDestroySVM(svm); ////
                    }
                }
            }
        }
    }

    /* save the best combination results in the CSV file */
    sprintf(csvFile->data[csvCount][0], "%s", "best_int_point_petector");
    sprintf(csvFile->data[csvCount][1], "%s", "best_local_feat_extractor");
    sprintf(csvFile->data[csvCount][2], "%s", "best_dict_estimator");
    sprintf(csvFile->data[csvCount][3], "%s", "best_cod_func");
    sprintf(csvFile->data[csvCount][4], "%s", "best_default_params");
    sprintf(csvFile->data[csvCount][5], "%s", "best_params");
    sprintf(csvFile->data[csvCount][6], "%s", "best_optm_cross_val_acc");
    sprintf(csvFile->data[csvCount][7], "%s", "best_test_acc");
    sprintf(csvFile->data[csvCount][8], "%s", "mean_eval_time");
    csvCount++;

    sprintf(csvFile->data[csvCount][0], "%s", csvFile->data[maxTestAccIdx][0]);
    sprintf(csvFile->data[csvCount][1], "%s", csvFile->data[maxTestAccIdx][1]);
    sprintf(csvFile->data[csvCount][2], "%s", csvFile->data[maxTestAccIdx][2]);
    sprintf(csvFile->data[csvCount][3], "%s", csvFile->data[maxTestAccIdx][3]);
    sprintf(csvFile->data[csvCount][4], "%s", csvFile->data[maxTestAccIdx][4]);
    sprintf(csvFile->data[csvCount][5], "%s", csvFile->data[maxTestAccIdx][5]);
    sprintf(csvFile->data[csvCount][6], "%s", csvFile->data[maxTestAccIdx][6]);
    sprintf(csvFile->data[csvCount][7], "%s", csvFile->data[maxTestAccIdx][7]);
    sprintf(csvFile->data[csvCount][8], "%s", csvFile->data[maxTestAccIdx][8]);
    iftWriteCSV(csvFile, "optm_params.csv", ';');

    /* print the best combination overall */
    fprintf(stdout, "\n============================================================================================================================\n"); fflush(stdout);
    fprintf(stdout, "\n****************************************************************************************************************************\n"); fflush(stdout);
    fprintf(stdout, "Best combination overall ...\n"); fflush(stdout);
    fprintf(stdout, "****************************************************************************************************************************\n\n"); fflush(stdout);
    fprintf(stdout, "best_int_point_petector: %s\n", csvFile->data[maxTestAccIdx][0]);
    fprintf(stdout, "best_local_feat_extractor: %s\n", csvFile->data[maxTestAccIdx][1]);
    fprintf(stdout, "best_dict_estimator: %s\n", csvFile->data[maxTestAccIdx][2]);
    fprintf(stdout, "best_cod_func: %s\n", csvFile->data[maxTestAccIdx][3]);
    fprintf(stdout, "best_default_params: %s\n", csvFile->data[maxTestAccIdx][4]);
    fprintf(stdout, "best_params: %s\n", csvFile->data[maxTestAccIdx][5]);
    fprintf(stdout, "best_optm_cross_val_acc: %s\n", csvFile->data[maxTestAccIdx][6]);
    fprintf(stdout, "best_test_acc: %s\n", csvFile->data[maxTestAccIdx][7]);
    fprintf(stdout, "mean_eval_time: %s\n", csvFile->data[maxTestAccIdx][8]);

    iftFree(labels);
    iftDestroySampler(&trainSampler);
    iftDestroyFileSet(&trainImgFileSet);
    iftDestroyFileSet(&testImgFileSet);
    iftDestroySampler(&crossValidSampler);
    iftDestroyCSV(&csvFile);

    return 0;
}
