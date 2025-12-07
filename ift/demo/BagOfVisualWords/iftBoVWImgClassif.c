//
// Created by Cesar Castelo on Jan 17, 2019
//

#include <ift.h>

#define CLASSIF_SVM_LINEAR    0
#define CLASSIF_SVM_RBF       1
#define CLASSIF_OPF           2

#define NORM_DIM_RED_SIMPLE     0
#define NORM_DIM_RED_ZSCORE     1
#define NORM_DIM_RED_PCA        2
#define NORM_DIM_RED_SUP_PCA    3

void iftReadSamplingMask(iftDict *funcParams, iftFileSet **trainMasksFileset, iftFileSet **testMasksFileset, int nImgsTrainSet, int nImgsTestSet)
{
    if(iftDictContainKey("sampling_masks", funcParams, NULL)) {
        printf("\n--> Reading sampling mask for training and testing sets ... "); fflush(stdout);
        char *path = iftGetStrValFromDict("sampling_masks", funcParams);
        
        (*trainMasksFileset) = iftCreateFileSet(nImgsTrainSet);
        for(int f = 0; f < nImgsTrainSet; f++)
            (*trainMasksFileset)->files[f] = iftCreateFile(path);
        
        (*testMasksFileset) = iftCreateFileSet(nImgsTestSet);
        for(int f = 0; f < nImgsTestSet; f++)
            (*testMasksFileset)->files[f] = iftCreateFile(path);
        printf("OK\n");
    }
}

void iftReadImgMasks(iftDict *funcParams, iftFileSet **trainMasksFileset, iftFileSet **testMasksFileset)
{
    printf("\n--> Reading masks for training and testing sets ... "); fflush(stdout);
    iftDict *dictAux = iftGetDictFromDict("int_point_detec", funcParams);
    char *trainMasksDir = iftGetStrValFromDict("train_masks_dir", dictAux);
    char *testMasksDir = iftGetStrValFromDict("test_masks_dir", dictAux);
    
    (*trainMasksFileset) = iftLoadFileSetFromDirOrCSV(trainMasksDir, 1, true);
    (*testMasksFileset) = iftLoadFileSetFromDirOrCSV(testMasksDir, 1, true);
    printf("OK\n");
}

int main(int argc, char** argv)
{
   if(argc != 7) {
        iftError("\nUsage: iftBoVWImgClassif <...>\n"
                         "[1] input_dict: Filename of the BoVW structure\n"
                         "[2] train_fileset: Input fileset with training images for classification (features will be extracted using the dictionary)\n"
                         "[3] test_fileset: Input fileset with testing images for classification (features will be extracted using the dictionary)\n"
                         "[4] classification_method:\n"
                         "    0: SVM (linear)\n"
                         "    1: SVM (RBF)\n"
                         "    2: OPF (supervised)\n"
                         "[5] save_feat_vect: Save feature vectors as datasets [0: No, 1: Yes] \n"
                         "[6] perform_tsne: Perform t-SNE on the feature vectors [0: No, 1: Train, 2: Train/Test (separated)] \n",
                 "iftBoVWImgClassif.c");
    }

    /* read parameters */
    int classifMeth = atoi(argv[4]);
    bool saveFeatVectors = (bool)atoi(argv[5]);
    int performTSNE = atoi(argv[6]);
    bool saveImgIntPts = false;
    char fileName[2048];

    /* print input parameters */
    iftPrintSeparatorLineInTerminal('=');
    printf("--> Input parameters:\n"); 
    printf("- input_dict: %s\n", argv[1]);
    printf("- train_fileset: %s\n", argv[2]);
    printf("- test_fileset: %s\n", argv[3]);
    printf("- classification_method: %s\n", classifMeth==CLASSIF_SVM_LINEAR ? "SVM-Linear" : classifMeth==CLASSIF_SVM_RBF ? "SVM-RBF" : "OPF");
    printf("- save_feat_vect: %s\n", saveFeatVectors ? "Yes" : "No");
    printf("- perform_tsne: %s\n", performTSNE ? "Yes" : "No");

    /* read the image filesets */
    iftPrintSeparatorLineInTerminal('=');
    printf("--> Reading input data ...\n");
    char *bovwFilename = iftCopyString(argv[1]);
    iftFileSet *trainFileset = iftLoadFileSetFromDirOrCSV(argv[2], 1, true);
    iftFileSet *testFileset = iftLoadFileSetFromDirOrCSV(argv[3], 1, true);
    printf("Num. Training images: %d\n", (int)trainFileset->n); 
    printf("Num. Testing images: %d\n", (int)testFileset->n); 
    printf("Num. Classes: %d\n", iftFileSetLabelsNumber(trainFileset));

    /* read the BoVW struct */
    printf("\n--> Reading the Visual Dictionary ...\n"); 
    iftBagOfVisualWords *bovw = iftBovwRead(bovwFilename);
    int intPointDetector = bovw->intPointDetectorId;
    int localFeatExtractor = bovw->localFeatExtractorId;
    int dictEstimator = bovw->dictEstimatorId;
    int codFunc = bovw->codFuncId;
    iftBovwPrintChosenMethods(intPointDetector, localFeatExtractor, dictEstimator, codFunc);
    printf("Parameters: \n");
    iftDict *dictAux = iftCopyDict(bovw->funcParams);
    iftRemoveValFromDict("output_dir", dictAux);
    iftRemoveValFromDict("output_suffix", dictAux);
    iftRemoveValFromDict("sampling_masks", dictAux);
    iftDict *dictAux1 = iftGetDictFromDict("int_point_detec", dictAux);
    iftRemoveValFromDict("n_classes", dictAux1);
    iftRemoveValFromDict("train_masks_dir", dictAux1);
    iftRemoveValFromDict("test_masks_dir", dictAux1);
    iftDict *dictAux2 = iftGetDictFromDict("local_feat_extr", dictAux);
    iftRemoveValFromDict("centralized_imgs_dir", dictAux2);
    iftRemoveValFromDict("kernels_dir", dictAux2);
    iftPrintDictAsArray(dictAux);
    iftPrintSeparatorLineInTerminal('=');

    /* read sampling masks */
    iftFileSet *trainMasksFileset = NULL, *testMasksFileset = NULL;
    if(intPointDetector != BOVW_INT_POINT_DETECTOR_SUP_SPIX_ISF) {
        iftReadSamplingMask(bovw->funcParams, &trainMasksFileset, &testMasksFileset, trainFileset->n, testFileset->n);
    }
    else {
        // iftReadImgMasks(bovw->funcParams, &trainMasksFileset, &testMasksFileset);
        iftDict *dictAux = iftGetDictFromDict("int_point_detec", bovw->funcParams);
        iftInsertIntoDict("n_classes", iftFileSetLabelsNumber(trainFileset), dictAux);
    }
    
    bool batchProcess = false;
    if(codFunc == BOVW_COD_FUNC_HARD_ASGMT_BATCH || codFunc == BOVW_COD_FUNC_SOFT_ASGMT_BATCH)
        batchProcess = true;

    /* perform local batch centralization and create the kernel bank */
    if(localFeatExtractor == BOVW_LOCAL_FEAT_EXTRACTOR_CONV) {
        iftBovwApplyLocalBatchCentralization(bovw, trainFileset);
        iftBovwApplyLocalBatchCentralization(bovw, testFileset);
    }

    /* perform local batch centralization and compute multilayer features */
    if(localFeatExtractor == BOVW_LOCAL_FEAT_EXTRACTOR_CONV_MULTI_LAYER) {
        iftBovwApplyLocalBatchCentralization(bovw, trainFileset);
        // iftBovwComputeMultiLayerConvolutionalFeaturesBatch(bovw, trainFileset);
        iftBovwApplyLocalBatchCentralization(bovw, testFileset);
        // iftBovwComputeMultiLayerConvolutionalFeaturesBatch(bovw, testFileset);
    }

    /* computing the feature vectors */
    iftImage *img0 = iftReadImageByExt(trainFileset->files[0]->path);
    iftImage *imgMask0 = NULL;
    if(trainMasksFileset != NULL) imgMask0 = iftReadImageByExt(trainMasksFileset->files[0]->path);
    iftRoiArray *roi0 = bovw->intPointDetector(img0, imgMask0, trainFileset->files[0], false, bovw->funcParams);
    timer *t1 = iftTic();

    printf("\n--> Computing the feature vectors for the training set...\n");
    int nVisWords = iftBovwGetNumbVisualWords(bovw);
    int nFeatsPerWord = iftBovwGetNumbFeatsPerWord(bovw);

    if(iftBovwIsFisherVectorsEncoding(bovw->codFuncId))
        printf("- fisher vectors encoding -> numb. clusters: %d, numb. local feats: %d -> %d final feats per image\n",
            nVisWords, nFeatsPerWord, nVisWords+2*nVisWords*nFeatsPerWord);
    else
        printf("- bovw encoding -> numb. visual words: %d, numb. feats per word: %d -> %d final feats per image\n",
            nVisWords, nFeatsPerWord, nVisWords);

    iftDataSet *Ztrain = NULL;
    if(batchProcess) {
        int batchSize = iftBovwComputeBatchSize(nVisWords, nFeatsPerWord, roi0->n, 0.95);
        printf("- batch size: %d\n", iftMin(batchSize, (int)trainFileset->n));
        Ztrain = iftBovwComputeFeatVectsFromImgFileSetBatch(bovw, trainFileset, trainMasksFileset, batchSize, saveImgIntPts);
    }
    else {
        Ztrain = iftBovwComputeFeatVectsFromImgFileSet(bovw, trainFileset, trainMasksFileset, saveImgIntPts);
    }

    printf("- applying standardization ... "); fflush(stdout);
    iftSetStatus(Ztrain, IFT_TRAIN);
    iftNormalizeDataSetByZScoreInPlace(Ztrain, NULL);
    printf("OK\n");
    if(iftBovwIsFisherVectorsEncoding(bovw->codFuncId)) {
        printf("- applying power normalization and L2 normalization ... "); fflush(stdout);
        iftPowerNormalizeDataSetInPlace(Ztrain, 0.5);
        iftNormalizeDataSetByL2NormInPlace(Ztrain);
        printf("OK\n");
    }
    
    printf("\n--> Computing the feature vectors for the testing set: \n"); 
    if(iftBovwIsFisherVectorsEncoding(bovw->codFuncId))
        printf("- fisher vectors encoding -> numb. clusters: %d, numb. local feats: %d -> %d final feats per image\n",
            nVisWords, nFeatsPerWord, nVisWords+2*nVisWords*nFeatsPerWord);
    else
        printf("- bovw encoding -> numb. visual words: %d, numb. feats per word: %d -> %d final feats per image\n",
            nVisWords, nFeatsPerWord, nVisWords);

    iftDataSet *Ztest = NULL;
    if(batchProcess) {
        int batchSize = iftBovwComputeBatchSize(nVisWords, nFeatsPerWord, roi0->n, 0.95);
        printf("- batch size: %d\n", iftMin(batchSize, (int)testFileset->n));
        Ztest = iftBovwComputeFeatVectsFromImgFileSetBatch(bovw, testFileset, testMasksFileset, batchSize, saveImgIntPts);
    }
    else {
        Ztest = iftBovwComputeFeatVectsFromImgFileSet(bovw, testFileset, testMasksFileset, saveImgIntPts);
    }
    
    printf("- applying standardization ... "); fflush(stdout);
    iftSetStatus(Ztest, IFT_TEST);
    iftNormalizeDataSetByZScoreInPlace(Ztest, &Ztrain->fsp);
    printf("OK\n");
    if(iftBovwIsFisherVectorsEncoding(bovw->codFuncId)) {
        printf("- applying power normalization and L2 normalization ... "); fflush(stdout);
        iftPowerNormalizeDataSetInPlace(Ztest, 0.5);
        iftNormalizeDataSetByL2NormInPlace(Ztest);
        printf("OK\n");
    }

    float featVectExtrTime = iftCompTime(t1, iftToc())/1000.0;
    printf("\nTime to compute the training and testing feature vectors: %s\n", iftFormattedTime(featVectExtrTime*1000.0)); 

    /* remove extra keys from dict */
    if(intPointDetector == BOVW_INT_POINT_DETECTOR_SUP_SPIX_ISF) {
        iftDict *dictAux = iftGetDictFromDict("int_point_detec", bovw->funcParams);
        iftRemoveValFromDict("n_classes", dictAux);
    }

    /* perform training and classification with the chosen classifier */
    iftDataSet *Zfinal = iftMergeDataSets(Ztrain, Ztest);
    iftDict *dictAux3 = iftGetDictFromDict("classif_met", bovw->funcParams);

    timer *t2 = iftTic();
    if(classifMeth == CLASSIF_SVM_LINEAR || classifMeth == CLASSIF_SVM_RBF) {
        int svmType;
        char *multiclass = iftGetStrValFromDict("multiclass", dictAux3);
        int svmMulticlass = iftCompareStrings(multiclass, "ovo") ? IFT_OVO : IFT_OVA;
        char *kernelizeStr = iftGetStrValFromDict("kernelize", dictAux3);
        int kernelize = iftCompareStrings(kernelizeStr, "yes") ? 1 : 0;
        float C = iftGetDblValFromDict("c", dictAux3);

        if(classifMeth == CLASSIF_SVM_LINEAR) {
            printf("\n--> Classifying the testing set with SVM-Linear (%s) ...\n", multiclass);
            svmType = IFT_LINEAR;
        }
        if(classifMeth == CLASSIF_SVM_RBF) {
            printf("\n--> Classifying the testing set with SVM-RBF (%s)...\n", multiclass);
            svmType = IFT_RBF;
        }

        iftSVM* svm = iftCreateSVM(svmType, svmMulticlass, C, 1.0/(float)Zfinal->nfeats);

        /* kernelization is implemented only for the Linear SVM */
        if (kernelize && svmType == IFT_LINEAR) {
            uchar traceNormalize = 0;
            svm->kernelization = kernelize;
            float ktrace;
            iftDataSet *Zaux = iftKernelizeDataSet(Zfinal, Zfinal, svmType, traceNormalize, &ktrace);
            if (svmMulticlass == IFT_OVO)
                iftSVMTrainOVO(svm, Zaux, Zfinal);
            else
                iftSVMTrainOVA(svm, Zaux, Zfinal);
            iftDestroyDataSet(&Zaux);
        }
        else {
            iftSVMTrain(svm, Zfinal);
        }

        iftSVMClassify(svm, Zfinal, IFT_TEST);
        iftDestroySVM(svm); ////
    }
    else if(classifMeth == CLASSIF_OPF) {
        printf("\n--> Classifying the testing set with Supervised OPF ...\n"); 
        iftCplGraph *graph = iftCreateCplGraph(Zfinal);
        iftSupTrain2(graph);
        iftClassify(graph, Zfinal);
        iftDestroyCplGraph(&graph);
    }
    double testKappa = iftCohenKappaScore(Zfinal);
    double testTruePos = iftTruePositives(Zfinal);
    iftDblArray *testTruePosPerClass = iftFloatArrayToDblArray(iftTruePositivesByClass(Zfinal));

    float classifTime = iftCompTime(t2, iftToc())/1000.0;
    printf("Cohen Kappa: %f\n", testKappa); 
    printf("True positives: %f\n", testTruePos);
    printf("True positives per class:\n");
    int *nSampPerClass = iftCountSamplesPerClassDataSet(Zfinal);
    for(int c = 1; c <= Zfinal->nclasses; c++)
        printf("- class %d: %f (%d samples)\n", c, testTruePosPerClass->val[c], nSampPerClass[c]);
    printf("Time to perform classification: %s\n", iftFormattedTime(classifTime*1000.0)); 
    
    /* save results in a CSV file */
    char *resultsFilename = iftGetStrValFromDict("results_filename", bovw->funcParams);
    iftDict *resultsJson = iftReadJson(resultsFilename);
    iftInsertIntoDict("classif_met", classifMeth == CLASSIF_SVM_LINEAR ? "svm-linear" : classifMeth == CLASSIF_SVM_RBF ? "svm-rbf" : "opf", resultsJson);
    iftInsertIntoDict("func_params", bovw->funcParams, resultsJson);
    iftInsertIntoDict("feat_vect_extr_time", featVectExtrTime, resultsJson);
    iftInsertIntoDict("classif_time", classifTime, resultsJson);
    iftInsertIntoDict("kappa", testKappa, resultsJson);
    iftInsertIntoDict("true_pos", testTruePos, resultsJson);
    iftInsertIntoDict("true_pos_per_class", testTruePosPerClass, resultsJson);
    iftWriteJson(resultsJson, resultsFilename);
    iftDestroyDict(&resultsJson);

    char *outputDir = iftGetStrValFromDict("output_dir", bovw->funcParams);
    char *outSuffix = iftGetStrValFromDict("output_suffix", bovw->funcParams);

    /* save the feature vectors */
    if(saveFeatVectors) {
        printf("\n--> Saving the feature vectors ...\n"); 
        sprintf(fileName, "%s/feat_vect_train_%s.zip", outputDir, outSuffix);
        iftWriteDataSet(Ztrain, fileName);
        sprintf(fileName, "%s/feat_vect_test_%s.zip", outputDir, outSuffix);
        iftWriteDataSet(Ztest, fileName);
    }

    /* projection with t-SNE */
    if(performTSNE) {
        printf("\n--> Creating the projection with t-SNE ...\n");
        iftDataSet *Ztrain2D = iftDimReductionByTSNE(Ztrain, 2, 40, 1000);
        iftImage *trainImg2D = iftDraw2DFeatureSpace(Ztrain2D, IFT_CLASS, IFT_ALL);
        iftColorTable *ctb = iftCategoricalColorTable(Ztrain->nclasses);
        iftDrawColorLegend(trainImg2D, ctb, 0, Ztrain->nclasses-1, "bottom-right", 15);
        sprintf(fileName, "%s/feat_vect_train_%s.png", outputDir, outSuffix);
        iftWriteImageByExt(trainImg2D, fileName);
        iftDestroyDataSet(&Ztrain2D);
        iftDestroyImage(&trainImg2D);
        iftDestroyColorTable(&ctb);
    }

    if(performTSNE == 2) {
        iftDataSet *Ztest2D = iftDimReductionByTSNE(Ztest, 2, 40, 1000);
        iftImage *testImg2D = iftDraw2DFeatureSpace(Ztest2D, IFT_CLASS, IFT_ALL);
        iftColorTable *ctb = iftCategoricalColorTable(Ztest->nclasses);
        iftDrawColorLegend(testImg2D, ctb, 0, Ztest->nclasses-1, "bottom-right", 15);
        sprintf(fileName, "%s/feat_vect_test_%s.png", outputDir, outSuffix);
        iftWriteImageByExt(testImg2D, fileName);
        iftDestroyDataSet(&Ztest2D);
        iftDestroyImage(&testImg2D);
        iftDestroyColorTable(&ctb);
    }

    iftDestroyFileSet(&trainFileset);
    iftDestroyFileSet(&trainMasksFileset);
    iftDestroyFileSet(&testFileset);
    iftDestroyFileSet(&testMasksFileset);
    iftDestroyBovw(&bovw);
    iftDestroyDataSet(&Zfinal);
    // iftDestroyDataSet(&Ztrain);
    // iftDestroyDataSet(&Ztest);

    iftPrintSeparatorLineInTerminal('=');
    return 0;
}
