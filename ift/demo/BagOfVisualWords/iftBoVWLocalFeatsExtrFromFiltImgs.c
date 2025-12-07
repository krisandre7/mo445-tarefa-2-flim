//
// Created by Cesar Castelo on Jan 24, 2019
//

#include <ift.h>

int main(int argc, char** argv)
{
    if(argc != 10) {
        iftError("\nUsage: iftBoVWLocalFeatsExtrFromFiltImgs <...>\n"
                    "[1] img_dir: Input directory containing the filtered images (PNG format)\n"
                    "[2] nImgs: Number of images\n"
                    "[3] nBands: Number of bands\n"
                    "[4] vect_constr: Method to construct the feature vector ['pixel_vals_in_patch','mean_val_in_patch','bic_in_patch']\n"
                    "[5] int_points_file: File containing the interest points (iftRoiArray format)\n"
                    "[6] n_bins_per_band: Number of bins per band (only if vect_constr='bic_in_patch', -1 otherwise)\n"
                    "[7] output_dirname: Output folder name\n"
                    "[8] output_suffix: Suffix to be added to the output files\n"
                    "[9] verbose: Messages printing [0: No, 1: Partial, 2: Full]\n",
                 "iftBoVWLocalFeatsExtrFromFiltImgs.c");
    }

    /* read parameters */
    char *imgDir = iftCopyString(argv[1]);
    int nImgs = atoi(argv[2]);
    int nBands = atoi(argv[3]);
    char *vectConstr = iftCopyString(argv[4]);
    char *intPointsFile = iftCopyString(argv[5]);
    int nBinsPerBand = atoi(argv[6]);
    char *outputDirname = iftCopyString(argv[7]);
    char *outSuffix = iftCopyString(argv[8]);
    int verbose = atoi(argv[9]);

    /* print input parameters */
    if(verbose == 2) {
        iftPrintSeparatorLineInTerminal('=');
        printf("--> Input parameters:\n"); 
        printf("- img_dir: %s\n", imgDir);
        printf("- n_imgs: %d\n", nImgs);
        printf("- n_bands: %d\n", nBands);
        printf("- vect_constr: %s\n", vectConstr);
        printf("- int_points_file: %s\n", intPointsFile);
        printf("- nBinsPerBand: %d\n", nBinsPerBand);
        printf("- output_dirname: %s\n", outputDirname);
        printf("- output_suffix: %s\n", outSuffix);
    }

    /* read the image fileset */
    if(verbose == 2) {
        iftPrintSeparatorLineInTerminal('=');
        printf("--> Reading input data ...\n");
    }

    char filename[2048];
    iftMImage **mimgSet = (iftMImage**)iftAlloc(nImgs, sizeof(iftMImage*));
    sprintf(filename, "%s/img_1_band_1.png", imgDir);
    iftImage *img0 = iftReadImageByExt(filename);
    for(int i = 0; i < nImgs; i++) {
        mimgSet[i] = iftCreateMImage(img0->xsize, img0->ysize, img0->zsize, nBands);
        for(int b = 0; b < nBands; b++) {
            sprintf(filename, "%s/img_%d_band_%d.png", imgDir, i+1, b+1);
            iftImage *img = iftReadImageByExt(filename);
            for(int p = 0; p < img->n; p++)
                mimgSet[i]->val[p][b] = img->val[p];
            iftDestroyImage(&img);
        }
    }

    /* read the interest points */
    iftRoiArray *roiArray = iftReadRoiArray(intPointsFile);
    int nRois = roiArray->n, nPtsRoi = roiArray->val[0]->n;

    /* build the local feature vectors */
    iftMatrix *localFeats = NULL;
    if(iftCompareStrings(vectConstr, "pixel_vals_in_patch")) {
        printf("- Building the feature vectors (pixel vals in patch) ...\n");
        localFeats = iftCreateMatrix(nPtsRoi*nBands, nImgs*nRois);
        for(int i = 0; i < nImgs; i++) {
            printf("  Image: %d/%d\r", i+1, nImgs); fflush(stdout);
            for(int p = 0; p < nRois; p++) {
                for(int b = 0; b < nBands; b++) {
                    for(int q = 0; q < nPtsRoi; q++) {
                        iftVoxel v = roiArray->val[p]->val[q];
                        int q1 = iftMGetVoxelIndex(mimgSet[i], v);
                        iftMatrixElem(localFeats, b*nPtsRoi+q, i*nRois+p) = mimgSet[i]->val[q1][b];
                    }
                }
            }
        }
    }
    else if(iftCompareStrings(vectConstr, "mean_val_in_patch")) {
        printf("- Building the feature vectors (mean val in patch) ...\n");
        localFeats = iftCreateMatrix(nBands, nImgs*nRois);
        for(int i = 0; i < nImgs; i++) {
            printf("  Image: %d/%d\r", i+1, nImgs); fflush(stdout);
            for(int p = 0; p < nRois; p++) {
                for(int b = 0; b < nBands; b++) {
                    float mean = 0;
                    for(int q = 0; q < nPtsRoi; q++) {
                        iftVoxel v = roiArray->val[p]->val[q];
                        int q1 = iftMGetVoxelIndex(mimgSet[i], v);
                        mean += mimgSet[i]->val[q1][b];
                    }
                    iftMatrixElem(localFeats, b, i*nRois+p) = mean / (nBands*nPtsRoi);
                }
            }
        }
    }
    else if(iftCompareStrings(vectConstr, "bic_in_patch")) {
        printf("- Building the feature vectors (BIC encoding in patch) ...\n");
        localFeats = iftCreateMatrix(nBands*nBinsPerBand*2, nImgs*nRois);
        for(int i = 0; i < nImgs; i++) {
            printf("  Image: %d/%d\r", i+1, nImgs); fflush(stdout);
            for(int p = 0; p < nRois; p++) {
                iftMImage* mimgRoi = iftMExtractRoiNoBkgd(mimgSet[i], roiArray->val[p]);
                iftFeatures *feat = iftMExtractBIC(mimgRoi, nBinsPerBand);
                iftCopyFloatArray(iftMatrixRowPointer(localFeats, i*nRois+p), feat->val, feat->n);
                iftDestroyMImage(&mimgRoi);
                iftDestroyFeatures(&feat);
            }
        }
    }
    printf("\n");

    sprintf(filename, "%s/local_feats_%s.npy", outputDirname, outSuffix);
    iftWriteMatrix(localFeats, filename);

    /* free memory */
    for(int i = 0; i < nImgs; i++)
        iftDestroyMImage(&mimgSet[i]);
    iftFree(mimgSet);
    iftDestroyImage(&img0);
    iftDestroyRoiArray(&roiArray);
    iftDestroyMatrix(&localFeats);

    return 0;
}
