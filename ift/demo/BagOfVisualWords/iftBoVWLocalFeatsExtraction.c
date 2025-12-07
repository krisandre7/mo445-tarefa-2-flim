//
// Created by Cesar Castelo on Apr 20, 2018, modified Nov 26, 2019
//

#include <ift.h>

int tsnePerplexity = 20, tsneNumIter = 1000;

int main(int argc, char** argv)
{
    if(argc != 11) {
        iftError("\nUsage: iftBoVWLocalFeatsExtraction <...>\n"
                    "[1] img_fileset: Input fileset with the images for local feature extraction\n"
                    "[2] img_masks_fileset: Input fileset with the images' masks for local feature extraction (-1 if not using masks)\n"
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
                    "[5] func_params_json: JSON file containing the function params (or the JSON structure itself as string)\n"
                    "[6] save_patches: Whether or not to save the patches that will be extracted from the images [0: No, 1: Yes]\n"
                    "[7] perform_tsne: Whether or not to perform t-SNE projection [0-No, 1-Yes]\n"
                    "[8] save_extra_info: Whether or not to save the features and the labels as NumPy arrays [0: No, 1: Yes]\n"
                    "[9] output_dir_basename: Basename for the output folder\n"
                    "[10] output_suffix: Suffix to be added to the output files\n",
                 "iftBoVWLocalFeatsExtraction.c");
    }

    /* read parameters */
    int intPointDetector = atoi(argv[3]);
    int localFeatExtractor = atoi(argv[4]);
    char *funcParamsJson = iftCopyString(argv[5]);
    bool savePatches = (bool)atoi(argv[6]);
    bool performTsne = (bool)atoi(argv[7]);
    bool saveExtraInfo = (bool)atoi(argv[8]);
    char *outputDirBasename = iftCopyString(argv[9]);
    char *outSuffix = iftCopyString(argv[10]);
    bool saveImgIntPts = false;
    int distFunc = 1;

    /* print input parameters */
    iftPrintSeparatorLineInTerminal('=');
    printf("--> Input parameters:\n"); 
    printf("- img_fileset: %s\n", argv[1]);
    printf("- img_masks_fileset: %s\n", !iftCompareStrings(argv[2], "-1") ? argv[2] : "Not using masks");
    printf("- func_params_json: %s\n", iftCompareStrings(iftFileExt(funcParamsJson), ".json") ? funcParamsJson : "JSON structure sent as text");
    printf("- save_patches: %s\n", savePatches ? "Yes" : "No");
    printf("- perform_tsne: %s\n", performTsne ? "Yes" : "No");
    printf("- output_dir_basename: %s\n", outputDirBasename);
    printf("- output_suffix: %s\n", outSuffix);

    /* read the image fileset */
    iftPrintSeparatorLineInTerminal('=');
    printf("--> Reading input data ...\n"); 
    iftFileSet *imgFileset = iftLoadFileSetFromDirOrCSV(argv[1], 1, true);

    iftFileSet *imgMasksFileset = NULL;
    if(!iftCompareStrings(argv[2], "-1"))
        imgMasksFileset = iftLoadFileSetFromDirOrCSV(argv[2], 1, true);

    /* read JSON file with function params */
    iftDict *funcParams = NULL;
    if(iftCompareStrings(iftFileExt(funcParamsJson), ".json"))
        funcParams = iftReadJson(funcParamsJson);
    else
        funcParams = iftDictFromJsonString(funcParamsJson);

    /* print chosen methods and params */
    printf("Num. Images for local feature extraction: %d\n", (int)imgFileset->n);
    printf("Num. Classes: %d\n", iftFileSetLabelsNumber(imgFileset));
    iftBovwPrintChosenMethods(intPointDetector, localFeatExtractor, IFT_NIL, IFT_NIL);
    printf("Parameters: \n"); 
    iftPrintDictAsArray(funcParams);
    iftPrintSeparatorLineInTerminal('=');
    
    /* create the BoVW struct */
    iftBagOfVisualWords *bovw = iftCreateBovw();
    iftBovwSetFunctionPointers(bovw, intPointDetector, localFeatExtractor, IFT_NIL, IFT_NIL, distFunc, funcParams);

    /* create the output dir */
    char *outputDir = iftConcatStrings(5, outputDirBasename, "_", iftBovwIntPointDetName(intPointDetector, false), "_",
        iftBovwLocalFeatExtrName(localFeatExtractor, false));
    if(!iftDirExists(outputDir))
        iftMakeDir(outputDir);

    iftInsertIntoDict("output_suffix", outSuffix, bovw->funcParams);
    iftInsertIntoDict("output_dir", outputDir, bovw->funcParams);

    /* create the directory for the image patches (if it is necessary) */
    char *patchDirPath = NULL;
    if(savePatches)
        patchDirPath = iftBovwCreateDirectoryForPatches(bovw);

    /* create the joint mask (if imgMasksFileset is set) */
    if(imgMasksFileset) {
        if(intPointDetector == BOVW_INT_POINT_DETECTOR_SUP_SPIX_ISF) {
            iftBovwCreateSamplingMasksPerClassISF(bovw, imgFileset, imgMasksFileset);
            iftDict *dictAux = iftGetDictFromDict("int_point_detec", bovw->funcParams);
            iftInsertIntoDict("n_classes", iftFileSetLabelsNumber(imgFileset), dictAux);
        }
        else {
            iftBovwCreateJointSamplingMask(bovw, &imgMasksFileset);
        }
    }

    /* perform local batch centralization and create the kernel bank */
    if(localFeatExtractor == BOVW_LOCAL_FEAT_EXTRACTOR_CONV) {
        iftBovwApplyLocalBatchCentralization(bovw, imgFileset);
        iftBovwCreateRandomKernels(bovw, imgFileset);
    }

    /* perform local batch centralization, create the kernel bank and compute multilayer features */
    if(localFeatExtractor == BOVW_LOCAL_FEAT_EXTRACTOR_CONV_MULTI_LAYER) {
        iftBovwApplyLocalBatchCentralization(bovw, imgFileset);
        iftBovwCreateRandomKernelsMultiLayer(bovw, imgFileset);
        // iftBovwComputeMultiLayerConvolutionalFeaturesBatch(bovw, imgFileset);
    }

    /* compute the local features */
    bovw->dictEstimator = NULL; // we set the dictEstimator pointer to NULL to not execute the dictionary estimator during BoVW learning
    char filename[2048];
    printf("\n--> Extracting the local features ...\n"); 
    float bovwTime = iftBovwLearnImgClassif(bovw, imgFileset, imgMasksFileset, patchDirPath, saveImgIntPts);
    printf("\nTotal time to extract the local features: %s\n", iftFormattedTime(bovwTime*1000.0));
    iftDataSet *localFeats = bovw->localFeats;
    sprintf(filename, "%s/local_feats_%s.zip", outputDir, outSuffix);
    iftWriteDataSet(localFeats, filename);

    /* save extra info from the local features */
    if(saveExtraInfo) {
        /* local features as NumPy ndarray */
        sprintf(filename, "%s/local_feats_%s.npy", outputDir, outSuffix);
        iftWriteMatrix(localFeats->data, filename);
        
        /* image labels and int points per image */
        iftDict *extraInfoDict = iftCreateDict();
        iftIntArray *labels = iftCreateIntArray(imgFileset->n);
        for(int i = 0; i < imgFileset->n; i++)
            labels->val[i] = imgFileset->files[i]->label;
        iftInsertIntoDict("img_labels", labels, extraInfoDict);

        iftDict *dictAux = iftGetDictFromDict("int_point_detec", bovw->funcParams);
        char *intPointsFile = iftGetStrValFromDict("int_points_file", dictAux);
        iftInsertIntoDict("int_points_file", intPointsFile, extraInfoDict);
        iftInsertIntoDict("n_int_points_per_img", localFeats->nsamples/imgFileset->n, extraInfoDict);

        sprintf(filename, "%s/extra_info_%s.json", outputDir, outSuffix);
        iftWriteJson(extraInfoDict, filename);
        iftDestroyDict(&extraInfoDict);
    }

    /* projection with t-SNE */
    if(performTsne) {
        printf("\n--> Creating the projection with t-SNE ...\n");

        /* t-SNE projection using IFT_CLASS */
        printf("- projections with all the feature vectors (class) ... "); fflush(stdout);
        iftDataSet *localFeats2D = iftDimReductionByTSNE(localFeats, 2, tsnePerplexity, tsneNumIter);
        iftImage *img2D = iftDraw2DFeatureSpace(localFeats2D, IFT_CLASS, (uchar)IFT_ALL);
        iftColorTable *ctb = iftCategoricalColorTable(localFeats->nclasses);
        iftDrawColorLegend(img2D, ctb, 0, localFeats->nclasses-1, "bottom-right", 15);
        sprintf(filename, "%s/local_feats_%s.png", outputDir, outSuffix);
        iftWriteImageByExt(img2D, filename);
        iftDestroyDataSet(&localFeats2D);
        iftDestroyImage(&img2D);
        iftDestroyColorTable(&ctb);
        printf("OK\n");
    }

    iftDestroyBovw(&bovw);

    return 0;
}
