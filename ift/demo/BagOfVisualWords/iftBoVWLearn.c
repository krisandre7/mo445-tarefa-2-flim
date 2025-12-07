//
// Created by Cesar Castelo on Jan 17, 2019
//

#include <ift.h>

int main(int argc, char** argv)
{
    if(argc != 13) {
        iftError("\nUsage: iftBoVWLearn <...>\n"
                    "[1] dict_learn_fileset: Input fileset with the images for dictionary learning\n"
                    "[2] dict_learn_masks_fileset: Input fileset with the images' masks for dictionary learning (-1 if not using masks)\n"
                    "[3] int_point_detection_method: Method to detect interest points\n"
                    "    0: Random\n"
                    "    1: Grid\n"
                    "    2: Unsupervided superpixels by ISF\n"
                    "    3: Supervised superpixels by ISF\n"
                    "[4] feat_extraction_method: Method to extract the local features\n"
                    "    0: Raw pixel values\n"
                    "    1: Border/Interior Pixel Classification (BIC)\n"
                    "    2: Local Binary Patterns (LBP)\n"
                    "    3: BRIEF descriptor\n"
                    "    4: Convolutional features\n"
                    "    5: Multi Layer Convolutional features\n"
                    "    6: Deep features from mimage\n"
                    "[5] dict_estimation_method: Method to estimate the dictionary\n"
                    "    0: Unsup kMeans\n"
                    "    1: Sup k-means with ordering by pixel label\n"
                    "    2: Sup k-means with ordering by image class\n"
                    "    3: Sup k-means with ordering by image\n"
                    "    4: Sup k-means with ordering by position\n"
                    "    5: Sup k-means with ordering by image class and position\n"
                    "    6: Unsup OPF\n"
                    "    7: Sup OPF with ordering by pixel label\n"
                    "    8: Sup OPF with ordering by image class\n"
                    "    9: Sup OPF with ordering by image\n"
                    "    10: Sup OPF with ordering by position\n"
                    "    11: Sup OPF with ordering by image class and position\n"
                    "    12: Sup Manual with ordering by image class\n"
                    "    13: Sup Manual with ordering by position\n"
                    "    14: Sup Manual with ordering by image class and position\n"
                    "[6] coding_method: Method to perform coding operation\n"
                    "    0: Hard assignment clustering\n"
                    "    1: Soft assignment clustering\n"
                    "    2: Hard assignment clustering with batch processing\n"
                    "    3: Soft assignment clustering with batch processing\n"
                    "    4: 2nd order statistics (fisher vectors)\n"
                    "[7] func_params_json: JSON file containing the function params (or the JSON structure itself as string)\n"
                    "[8] save_patches: Save the patches that will be extracted from the images [0: No, 1: Yes]\n"
                    "[9] save_dict_dataset: Save the visual dictionary as a separate dataset\n"
                    "[10] sel_words_json: JSON file containing the selected visual words (only when using manual dictionary estimation, -1 otherwise)\n"
                    "[11] output_dir_basename: Basename for the output folder\n"
                    "[12] output_suffix: Suffix to be added to the output files\n",
                 "iftBoVWLearn.c");
    }

    /* read parameters */
    int intPointDetector = atoi(argv[3]);
    int localFeatExtractor = atoi(argv[4]);
    int dictEstimator = atoi(argv[5]);
    int codFunc = atoi(argv[6]);
    char *funcParamsJson = iftCopyString(argv[7]);
    bool savePatches = (bool)atoi(argv[8]);
    bool saveDictAsDS = (bool)atoi(argv[9]);
    char *selWordsJson = iftCopyString(argv[10]);
    char *outputDirBasename = iftCopyString(argv[11]);
    char *outSuffix = iftCopyString(argv[12]);
    bool saveImgIntPts = false;
    int distFunc = 1;

    /* print input parameters */
    iftPrintSeparatorLineInTerminal('=');
    printf("--> Input parameters:\n"); 
    printf("- dict_learn_fileset: %s\n", argv[1]);
    printf("- dict_learn_masks_fileset: %s\n", !iftCompareStrings(argv[2], "-1") ? argv[2] : "Not using masks");
    printf("- func_params_json: %s\n", iftCompareStrings(iftFileExt(funcParamsJson), ".json") ? funcParamsJson : "JSON structure sent as text");
    printf("- save_patches: %s\n", savePatches ? "Yes" : "No");
    printf("- save_dict_dataset: %s\n", saveDictAsDS ? "Yes" : "No");
    if(!iftCompareStrings(selWordsJson, "-1")) printf("- sel_words_json: %s\n", selWordsJson);
    printf("- output_dir_basename: %s\n", outputDirBasename);
    printf("- output_suffix: %s\n", outSuffix);

    if(saveDictAsDS && iftBovwIsFisherVectorsEncoding(codFunc))
        iftError("It is not possible to save the dictionary as a Dataset when using Fisher Vectors encoding", "iftBoVWLearn.c");

    /* read the image fileset */
    iftPrintSeparatorLineInTerminal('=');
    printf("--> Reading input data ...\n"); 
    iftFileSet *dictLearnFileset = iftLoadFileSetFromDirOrCSV(argv[1], 1, true);

    iftFileSet *dictLearnMasksFileset = NULL;
    if(!iftCompareStrings(argv[2], "-1"))
        dictLearnMasksFileset = iftLoadFileSetFromDirOrCSV(argv[2], 1, true);

    /* read JSON file with function params */
    iftDict *funcParams = NULL;
    if(iftCompareStrings(iftFileExt(funcParamsJson), ".json"))
        funcParams = iftReadJson(funcParamsJson);
    else
        funcParams = iftDictFromJsonString(funcParamsJson);

    /* print chosen methods and params */
    printf("Num. Images for dictionary learning: %d\n", (int)dictLearnFileset->n);
    printf("Num. Classes: %d\n", iftFileSetLabelsNumber(dictLearnFileset));
    iftBovwPrintChosenMethods(intPointDetector, localFeatExtractor, dictEstimator, codFunc);
    printf("Parameters: \n"); 
    iftPrintDictAsArray(funcParams);
    iftPrintSeparatorLineInTerminal('=');

    /* create the BoVW struct */
    iftBagOfVisualWords *bovw = iftCreateBovw();
    iftBovwSetFunctionPointers(bovw, intPointDetector, localFeatExtractor, dictEstimator, codFunc, distFunc, funcParams);

    /* create the output dir */
    char *outputDir = iftConcatStrings(9, outputDirBasename, "_", iftBovwIntPointDetName(intPointDetector, false), "_",
        iftBovwLocalFeatExtrName(localFeatExtractor, false), "_", iftBovwDictEstimName(dictEstimator, false), "_",
        iftBovwCodFuncName(codFunc, false));
    if(!iftDirExists(outputDir))
        iftMakeDir(outputDir);

    iftInsertIntoDict("output_suffix", outSuffix, bovw->funcParams);
    iftInsertIntoDict("output_dir", outputDir, bovw->funcParams);

    /* create the directory for the image patches (if it is necessary) */
    char *patchDirPath = NULL;
    if(savePatches)
        patchDirPath = iftBovwCreateDirectoryForPatches(bovw);

    /* create the joint mask (if dictLearnMasksFileset is set) */
    if(dictLearnMasksFileset) {
        if(intPointDetector == BOVW_INT_POINT_DETECTOR_SUP_SPIX_ISF) {
            iftBovwCreateSamplingMasksPerClassISF(bovw, dictLearnFileset, dictLearnMasksFileset);
            iftDict *dictAux = iftGetDictFromDict("int_point_detec", bovw->funcParams);
            iftInsertIntoDict("n_classes", iftFileSetLabelsNumber(dictLearnFileset), dictAux);
        }
        else {
            iftBovwCreateJointSamplingMask(bovw, &dictLearnMasksFileset);
        }
    }

    /* perform local batch centralization and create the kernel bank */
    if(localFeatExtractor == BOVW_LOCAL_FEAT_EXTRACTOR_CONV) {
        iftBovwApplyLocalBatchCentralization(bovw, dictLearnFileset);
        iftBovwCreateRandomKernels(bovw, dictLearnFileset);
    }

    /* perform local batch centralization, create the kernel bank and compute multilayer features */
    if(localFeatExtractor == BOVW_LOCAL_FEAT_EXTRACTOR_CONV_MULTI_LAYER) {
        iftBovwApplyLocalBatchCentralization(bovw, dictLearnFileset);
        iftBovwCreateRandomKernelsMultiLayer(bovw, dictLearnFileset);
        // iftBovwComputeMultiLayerConvolutionalFeaturesBatch(bovw, dictLearnFileset);
    }

    /* include the file with the manually selected centroids (if it is necessary)*/
    if(dictEstimator == BOVW_DICT_ESTIMATOR_SUP_MANUAL_IMG_CLASS ||
        dictEstimator == BOVW_DICT_ESTIMATOR_SUP_MANUAL_POSITION ||
        dictEstimator == BOVW_DICT_ESTIMATOR_SUP_MANUAL_IMG_CLASS_AND_POSITION) {
        iftDict *dictAux = iftGetDictFromDict("dict_estim", bovw->funcParams);
        iftInsertIntoDict("sel_cent_json_filename", selWordsJson, dictAux);
    }

    /* perform BoVW learning */
    printf("\n--> Learning the Visual Dictionary ...\n"); 
    float bovwTime = iftBovwLearnImgClassif(bovw, dictLearnFileset, dictLearnMasksFileset, patchDirPath, saveImgIntPts);
    printf("\nTotal time to perform BoVW learning: %s\n", iftFormattedTime(bovwTime*1000.0));

    if(bovw->orderHrchies) {
        iftDict *dictAux = iftGetDictFromDict("dict_estim", bovw->funcParams);
        iftRemoveValFromDict("local_feats_order_hrchy", dictAux);
    }

    /* add extra information for fisher vectors computation */
    if(iftBovwIsFisherVectorsEncoding(bovw->codFuncId)) {
        iftDict *dictAux = iftGetDictFromDict("cod_func", bovw->funcParams);
        iftInsertIntoDict("dict_estim_id", dictEstimator, dictAux);

        if(bovw->orderHrchies) {
            if(bovw->orderHrchies->n == 1) {
                int nClustModels = iftCountUniqueIntElems(iftMatrixRowPointer(bovw->localFeatsOrderHrchy, 0), bovw->localFeats->nsamples);
                iftInsertIntoDict("n_clust_models", nClustModels, dictAux);
            }

            if(bovw->orderHrchies->n == 2) {
                int nClustModels1 = iftCountUniqueIntElems(iftMatrixRowPointer(bovw->localFeatsOrderHrchy, 0), bovw->localFeats->nsamples);
                int nClustModels2 = iftCountUniqueIntElems(iftMatrixRowPointer(bovw->localFeatsOrderHrchy, 1), bovw->localFeats->nsamples);
                iftInsertIntoDict("n_clust_models_1", nClustModels1, dictAux);
                iftInsertIntoDict("n_clust_models_2", nClustModels2, dictAux);
            }
        }
    }

    /* save results in a JSON file */
    iftDict *resultsJson = iftCreateDict();
    iftInsertIntoDict("int_point_detector", iftBovwIntPointDetName(intPointDetector, false), resultsJson);
    iftInsertIntoDict("local_feat_extractor", iftBovwLocalFeatExtrName(localFeatExtractor, false), resultsJson);
    iftInsertIntoDict("dict_estimator", iftBovwDictEstimName(dictEstimator, false), resultsJson);
    iftInsertIntoDict("cod_func", iftBovwCodFuncName(codFunc, false), resultsJson);
    iftInsertIntoDict("n_classes", iftFileSetLabelsNumber(dictLearnFileset), resultsJson);
    iftInsertIntoDict("n_visual_words", iftBovwGetNumbVisualWords(bovw), resultsJson);
    iftInsertIntoDict("n_feats_per_word", bovw->localFeats->nfeats, resultsJson);
    iftInsertIntoDict("dict_learn_time", bovwTime, resultsJson);
    char fileName[2048];
    sprintf(fileName, "%s/results_%s.json", outputDir, outSuffix);
    iftWriteJson(resultsJson, fileName);
    iftDestroyDict(&resultsJson);
    iftInsertIntoDict("results_filename", fileName, bovw->funcParams);

    /* save the dictionary */
    printf("\n--> Saving the BoVW structure ... "); fflush(stdout);
    char *dictName = iftConcatStrings(3, "dict_", outSuffix, ".bovw");
    char *bovwFilename = iftJoinPathnames(2, outputDir, dictName);
    iftBovwWrite(bovw, bovwFilename);
    printf("OK\n");

    /* save the dictionary as a separate dataset (if it is necessary) */
    if(saveDictAsDS && !iftBovwIsFisherVectorsEncoding(bovw->codFuncId)) {
        iftDataSet *dict = (iftDataSet*)bovw->dict;
        iftAddStatus(dict, IFT_SUPERVISED);
        dictName = iftConcatStrings(3, "dict_", outSuffix, ".zip");
        bovwFilename = iftJoinPathnames(2, outputDir, dictName);
        iftWriteDataSet(dict, bovwFilename);
    }

    iftDestroyFileSet(&dictLearnFileset);
    iftDestroyFileSet(&dictLearnMasksFileset);
    iftDestroyBovw(&bovw);
    iftDestroyDict(&funcParams);

    iftPrintSeparatorLineInTerminal('=');
    return 0;
}
