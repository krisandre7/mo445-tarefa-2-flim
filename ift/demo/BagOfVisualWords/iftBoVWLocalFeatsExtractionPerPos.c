//
// Created by Cesar Castelo on Apr 20, 2018
//

#include <ift.h>

int kernelSize = 25*25, stride = 35, nKernels = 64, tsnePerplexity = 20, tsneNumIter = 1000, knnNumNeighb = 15;

void iftBovwCreateParamDicts(iftDict *funcParams, int intPointDetector, int localFeatExtractor)
{
    switch(intPointDetector) {
        case BOVW_INT_POINT_DETECTOR_RANDOM:
            iftInsertIntoDict("intPointDetec_nPoints", 20, funcParams);
            iftInsertIntoDict("intPointDetec_patchSizeX", (int)sqrt(kernelSize), funcParams);
            iftInsertIntoDict("intPointDetec_patchSizeY", (int)sqrt(kernelSize), funcParams);
            iftInsertIntoDict("intPointDetec_patchSizeZ", 1, funcParams);
            break;

        case BOVW_INT_POINT_DETECTOR_GRID:
            iftInsertIntoDict("intPointDetec_patchSizeX", (int)sqrt(kernelSize), funcParams);
            iftInsertIntoDict("intPointDetec_patchSizeY", (int)sqrt(kernelSize), funcParams);
            iftInsertIntoDict("intPointDetec_patchSizeZ", 1, funcParams);
            iftInsertIntoDict("intPointDetec_strideX", stride, funcParams);
            iftInsertIntoDict("intPointDetec_strideY", stride, funcParams);
            iftInsertIntoDict("intPointDetec_strideZ", 0, funcParams);
            break;

        case BOVW_INT_POINT_DETECTOR_SUPVOX_BOUNDARY:
            iftInsertIntoDict("intPointDetec_nSupervoxels", 900, funcParams);
            iftInsertIntoDict("intPointDetec_patchSize", 11, funcParams);
            break;
    }
    switch(localFeatExtractor) {
        case BOVW_LOCAL_FEAT_EXTRACTOR_RAW:
            iftInsertIntoDict("localFeatExtr_colSpaceIn", YCbCr_CSPACE, funcParams);
            iftInsertIntoDict("localFeatExtr_colSpaceOut", LABNorm_CSPACE, funcParams);
            break;

        case BOVW_LOCAL_FEAT_EXTRACTOR_BIC:
            iftInsertIntoDict("localFeatExtr_nBins", 8, funcParams);
            break;
        
        case BOVW_LOCAL_FEAT_EXTRACTOR_CONV:
            iftInsertIntoDict("localFeatExtr_nKernels", nKernels, funcParams);
            iftInsertIntoDict("localFeatExtr_kernelSize", kernelSize, funcParams);
            break;
    }
}

int main(int argc, char** argv)
{
    if(argc != 10) {
        iftError("\nUsage: iftBoVWLocalFeatsExtractionPerPos <...>\n"
                         "[1] train_fileset: Input fileset with training images\n"
                         "[2] train_masks_fileset: Input fileset with training images' masks (-1 if not using masks)\n"
                         "[3] detection_method: Method to detect interest points\n"
                         "    0: Random [BOVW_INT_POINT_DETECTOR_RANDOM]\n"
                         "    1: Grid [BOVW_INT_POINT_DETECTOR_GRID]\n"
                         "[4] extraction_method: Method to extract the local features\n"
                         "    0: Raw [BOVW_LOCAL_FEAT_EXTRACTOR_RAW]\n"
                         "    1: BIC [BOVW_LOCAL_FEAT_EXTRACTOR_BIC]\n"
                         "    2: LBP [BOVW_LOCAL_FEAT_EXTRACTOR_LBP]\n"
                         "    3: Convolution [BOVW_LOCAL_FEAT_EXTRACTOR_CONV]\n"
                         "[5] perform_clust: Perform clustering (per position)\n"
                         "[6] create_tsne Create t-SNE projection [0-No, 1-Yes]\n"
                         "[7] save_patches: Save the patches that will be extracted from the images [0: No, 1: Yes]\n"
                         "[8] output_dir_basename: Basename for the output folder\n"
                         "[9] output_suffix: Suffix to be added to the output files\n",
                 "iftBoVWLocalFeatsExtractionPerPos.c");
    }

    /* read parameters */
    int intPointDetector = atoi(argv[3]);
    int localFeatExtractor = atoi(argv[4]);
    bool performClust = (bool)atoi(argv[5]);
    bool createProj = (bool)atoi(argv[6]);
    bool savePatches = (bool)atoi(argv[7]);
    char *outputDirBasename = iftCopyString(argv[8]);
    char *outSuffix = iftCopyString(argv[9]);
    bool saveImgIntPts = false;
    int dictEstimator = BOVW_DICT_ESTIMATOR_SUP_OPF_POSITION; // this dict estimator is chosen to create the ordering hierarchy by position
    int distFunc = 1;

    /* read the image fileset */
    printf("============================================================================================================================\n"); 
    printf("--> Reading the image set ...\n"); 
    iftFileSet *trainFileset = iftLoadFileSetFromDirOrCSV(argv[1], 1, true);

    iftFileSet *trainMasksFileset = NULL;
    if(!iftCompareStrings(argv[2], "-1"))
        trainMasksFileset = iftLoadFileSetFromDirOrCSV(argv[2], 1, true);

    char *outputDir = iftConcatStrings(7, outputDirBasename, "_",
                                       iftBovwIntPointDetName(intPointDetector, false), "_",
                                       iftBovwLocalFeatExtrName(localFeatExtractor, false), "_",
                                       iftBovwDictEstimName(dictEstimator, false));
    if(!iftDirExists(outputDir))
        iftMakeDir(outputDir);

    char *patchDirPath = NULL;
    if(savePatches) {
        char patchStr[64]; sprintf(patchStr, "patch%d_stride%d", (int)sqrt(kernelSize), stride);
        patchDirPath = iftJoinPathnames(3, outputDir, "patches", patchStr);
        if(!iftDirExists(patchDirPath))
            iftMakeDir(patchDirPath);
    }
    
    printf("Num. Training images: %d\n", (int)trainFileset->n);
    printf("Num. Classes: %d\n", iftFileSetLabelsNumber(trainFileset));
    iftBovwPrintChosenMethods(intPointDetector, localFeatExtractor, dictEstimator, IFT_NIL);

    /* create the BoVW struct */
    iftBagOfVisualWords *bovw = iftCreateBovw();
    iftDict *funcParams = iftCreateDict();
    iftBovwCreateParamDicts(funcParams, intPointDetector, localFeatExtractor);
    iftBovwSetFunctionPointers(bovw, intPointDetector, localFeatExtractor, dictEstimator, IFT_NIL, distFunc, funcParams);
    printf("Parameters: \n"); 
    iftPrintDictAsArray(funcParams);
    printf("============================================================================================================================\n"); 

    /* if we are using convolution to extract the local features, we need to perform local batch centralization and create the kernel bank */
    iftImage *img0 = iftReadImageByExt(trainFileset->files[0]->path);
    int nBands = iftIsColorImage(img0) ? 3 : 1;
    int nImgs = trainFileset->n;

    if(localFeatExtractor == BOVW_LOCAL_FEAT_EXTRACTOR_CONV) {
        printf("\n--> Applying local batch centralization to the images ... "); fflush(stdout);
        char kernSizeStr[32]; sprintf(kernSizeStr, "kernel_%dx%d", (int)sqrt(kernelSize), (int)sqrt(kernelSize));
        char *centralizedImgsPath = iftJoinPathnames(4, outputDir, "centralized_imgs", "learning", kernSizeStr);
        
        if(!iftDirExists(centralizedImgsPath))
            iftMakeDir(centralizedImgsPath);
        centralizedImgsPath = iftAbsPathname(centralizedImgsPath);

        /* verify if the path of centralizaed images already contains the images */
        iftFileSet *fs = iftLoadFileSetFromDirOrCSV(centralizedImgsPath, 1, true);
        if(fs->n == nImgs) {
            printf("images already centralized!\n");
        }
        else {
            iftLocalBatchCentralization(trainFileset, kernelSize, centralizedImgsPath);
            printf("OK\n");
        }
        iftInsertIntoDict("localFeatExtr_centralizedImgsPath", centralizedImgsPath, bovw->funcParams);
        iftDestroyFileSet(&fs);

        /* create the random kernels */
        char *kernelsPath = iftJoinPathnames(2, outputDir, "kernels.npy");
        iftMatrix *kernels = iftRandomKernelBankAsMatrix(kernelSize, nBands, nKernels);
        iftWriteMatrix(kernels, kernelsPath);
        kernelsPath = iftAbsPathname(kernelsPath);
        iftDestroyMatrix(&kernels);
        iftInsertIntoDict("localFeatExtr_kernelsPath", kernelsPath, bovw->funcParams);
    }

    /* compute the local features */
    bovw->dictEstimator = NULL; // we set the dictEstimator pointer to NULL to not execute the dictionary estimator during BoVW learning
    char filename[2048];
    printf("\n--> Extracting the local features ...\n"); 
    float bovwTime = iftBovwLearnImgClassif(bovw, trainFileset, trainMasksFileset, patchDirPath, saveImgIntPts);
    printf("\nTotal time to extract the local features: %s\n", iftFormattedTime(bovwTime*1000.0));

    /* divide the dataset into an array of datasets per position */
    iftDataSet *localFeats = bovw->localFeats;
    int nPositions = localFeats->nsamples/nImgs; // we assume that we extract the same number of positions per image
    iftDataSet **localFeatsPerPos = (iftDataSet**)iftAlloc(nPositions, sizeof(iftDataSet*));
    
    for(int p = 0; p < nPositions; p++) {
        localFeatsPerPos[p] = iftCreateDataSet(nImgs, localFeats->nfeats);
        localFeatsPerPos[p]->nclasses = localFeats->nclasses;
        iftSetDistanceFunction(localFeatsPerPos[p], bovw->distFuncId);
        if(patchDirPath) {
            localFeatsPerPos[p]->ref_data_type = IFT_REF_DATA_FILESET;
            localFeatsPerPos[p]->ref_data = iftCreateFileSet(nImgs);
        }
    }

    iftIntArray *copied = iftCreateIntArray(nPositions);
    iftIntMatrix *localFeatsOrderHrchy = bovw->localFeatsOrderHrchy;
    for(int i = 0; i < localFeats->nsamples; i++) {
        int pos = iftMatrixElem(localFeatsOrderHrchy, i, 0) - 1; // pos starts in 1
        int s = copied->val[pos];
        iftCopySample(&localFeats->sample[i], &localFeatsPerPos[pos]->sample[s], localFeats->nfeats, true);
        localFeatsPerPos[pos]->sample[s].id = s;
        if(patchDirPath) {
            iftFileSet *fs1 = (iftFileSet*)localFeatsPerPos[pos]->ref_data;
            iftFileSet *fs2 = (iftFileSet*)localFeats->ref_data;
            fs1->files[s] = iftCopyFile(fs2->files[i]);
        }
        copied->val[pos]++;
    }

    /* perform clustering */
    if(performClust) {
        printf("\n--> Performing clustering per position ...\n");

        for(int p = 0; p < nPositions; p++) {
            printf("- position %d/%d (nsamples: %d) ...\r", p+1, nPositions, localFeatsPerPos[p]->nsamples); fflush(stdout);
            iftAddStatus(localFeatsPerPos[p], IFT_TRAIN);
            iftKnnGraph *graph = iftCreateKnnGraph(localFeatsPerPos[p], knnNumNeighb);
            iftFastUnsupTrain(graph, iftNormalizedCut);
        }
        printf("\n");
    }

    iftBoundingBoxArray *roi0 = bovw->intPointDetector(img0, NULL, trainFileset->files[0], false, bovw->funcParams);

    printf("\n--> Writing the datasets per position ...\n");
    for(int p = 0; p < nPositions; p++) {
        printf("- position %d/%d ...\r", p+1, nPositions); fflush(stdout);
        iftVoxel v = iftBoundingBoxCenterVoxel(roi0->val[p]);
        sprintf(filename, "%s/localfeats_x%04d_y%04d_%s.zip", outputDir, v.x, v.y, outSuffix);
        iftWriteDataSet(localFeatsPerPos[p], filename);
    }
    printf("\n");

    /* projection with t-SNE */
    if(createProj) {
        printf("\n--> Creating the projections with t-SNE ...\n"); 

        /* t-SNE projection using IFT_CLASS */
        for(int p = 0; p < nPositions; p++) {
            printf("- projections for every position (class) ... position %d/%d\r", p+1, nPositions); fflush(stdout);
            iftDataSet *localFeats2D = iftDimReductionByTSNE(localFeatsPerPos[p], 2, tsnePerplexity, tsneNumIter);
            iftImage *img2D = iftDraw2DFeatureSpace(localFeats2D, IFT_CLASS, (uchar)IFT_ALL);
            iftColorTable *ctb = iftCategoricalColorTable(localFeatsPerPos[p]->nclasses);
            iftDrawColorLegend(img2D, ctb, 0, localFeatsPerPos[p]->nclasses-1, "bottom-right", 15);
            iftVoxel v = iftBoundingBoxCenterVoxel(roi0->val[p]);
            sprintf(filename, "%s/localfeats_x%04d_y%04d_CLASS_%s.png", outputDir, v.x, v.y, outSuffix);
            iftWriteImageByExt(img2D, filename);
            iftDestroyDataSet(&localFeats2D);
            iftDestroyImage(&img2D);
            iftDestroyColorTable(&ctb);
        }
        printf("\n");

        if(performClust) {
            /* t-SNE projection using IFT_GROUP */
            for(int p = 0; p < nPositions; p++) {
                printf("- projections for every position (group) ... position %d/%d\r", p+1, nPositions); fflush(stdout);
                iftDataSet *localFeats2D = iftDimReductionByTSNE(localFeatsPerPos[p], 2, tsnePerplexity, tsneNumIter);
                iftImage *img2D = iftDraw2DFeatureSpace(localFeats2D, IFT_GROUP, (uchar)IFT_ALL);
                iftVoxel v = iftBoundingBoxCenterVoxel(roi0->val[p]);
                sprintf(filename, "%s/localfeats_x%04d_y%04d_GROUP_%s.png", outputDir, v.x, v.y, outSuffix);
                iftWriteImageByExt(img2D, filename);
                iftDestroyDataSet(&localFeats2D);
                iftDestroyImage(&img2D);
            }
        }
        printf("\n");
    }

    for(int p = 0; p < nPositions; p++)
        iftDestroyDataSet(&localFeatsPerPos[p]);
    iftFree(localFeatsPerPos);
    iftDestroyImage(&img0);
    iftDestroyBoundingBoxArray(&roi0);

    iftDestroyFileSet(&trainFileset);
    iftDestroyFileSet(&trainMasksFileset);
    iftDestroyBovw(&bovw);

    return 0;
}
