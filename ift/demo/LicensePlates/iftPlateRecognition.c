#include "ift.h"

#define BOX_SIZE_X 100
#define BOX_SIZE_Y 45

#define DESC_FREQ_X 35
#define DESC_FREQ_Y 15

#define MIN_CH 400
#define NORM_VAL_FEAT_CH 100

#define MIN_NUM_PIX_CHAR 35
#define MAX_NUM_PIX_CHAR 180

#define MIN_Y_CHAR_SIZE 12
#define MAX_Y_CHAR_SIZE 32
#define RATIO_CHAR 1.0

#define WRITE_TEMP_IMGS 0


/* Feature extraction functions */
iftImage *enhanceImage(iftImage *orig);
iftImage *enhanceImageFuzzy(iftImage *orig);
iftImage *countComponentsX(iftImage *orig);
iftImage *countComponentsY(iftImage *orig);
iftFeatures *getFreqFeatures(iftImage *compX, iftImage *compY, iftVoxel u, iftVoxel v);
float sumFeatureValues(iftFeatures *feat);
iftFeatures* computeLBP(iftImage *img);

/* Normalization function */
iftDataSet *iftNormalizeDataSetByGlobalMax(iftDataSet *Z);
iftDataSet *iftNormalizeDataSetByLocalMax(iftDataSet *Z);

/* Bounding box functions */
void getBoundingBoxCoord(iftImage *label, int val, iftVoxel *u, iftVoxel *v);
iftImage *createBoundingBox2DByCoord(iftImage *img, iftVoxel u, iftVoxel v);
void setBoundingBox(iftImage *label, iftVoxel u, iftVoxel v);
void addValueToBoundingBox(iftFImage *label, iftVoxel u, iftVoxel v, float value);

/* Train sample selection functions */
void selectTrainSamples(iftImage *orig, iftImage *label, iftImage *gradientImg, int nsamples, iftImage *compX, iftImage *compY, iftDataSet *Z, int indexZ);
int isCandidateSample(iftImage *compX, iftImage *compY, iftVoxel u, iftVoxel v);
int randInterval(int min, int max);
int isAvailableVoxel(iftImage *label, iftVoxel u, iftVoxel v);

/* Detect plates in new images */
iftImage *detectPlate(iftImage *orig, iftImage *label, iftSVM* svm, char *filename);
void computeMetrics(iftImage *testResult, iftImage *testLabel, int *tp, int *fp, int *fn, float *intersecCoef, float *intersecGT);
iftImage *filteringCandidates(iftImage *orig, iftImage *candidates, iftImage *label, char *filename);
iftImage *drawGrayMarkers(iftImage *label, int bgMarkerValue, iftImage *objMarker, iftAdjRel *A);
iftImage *getComponentBasedMarkers(iftImage *relabelThrImg, char *filename);

/* Register */
iftImage *correctPlate(iftImage *orig, iftImage *label);
iftImage* correctPlateF(iftImage *origImg, iftImage *filtImg);
iftPoint *iftTheFourPlateCorners(iftImage *plate);
iftImage *iftCorrectImage(iftImage *img, iftPoint *src_pt, iftPoint *dst_pt, int xsize, int ysize);
iftMatrix *iftCameraCorrectionMatrix(iftPoint *src, iftPoint *dst);
iftImage* iftPlateCharacterSeparation(iftImage* correctedPlate);
iftMatrix *iftInvertMatrixAw(iftMatrix *A);

// Character segmentation
iftImage* iftAwesomeCharacterSeparation(iftImage* correctedPlate);

/* Character recognition - ALPR program */
void opticalCharacterRecognition(iftImage* img, const char* filename);


int main(int argc, char *argv[]) {

    iftImage *trainOrig, *trainLabel, *testOrig, *testLabel, *gradientImg, *compX, *compY, *testResult, *filtResult, *plate;
    iftDataSet *Z, *Zn;
    int indexZ, samplesPerImage, tp, fp, fn;
    float intersecCoef, totalInterCoef, intersecGT, totalInterGT, precision, recall, accCharacter, meanNumCompChar;
    timer *t1 = NULL, *t2 = NULL;
    float time, timeOCR;
    char outfilePlate[100], outFileTest[100];
    FILE *f1;



    if (argc != 7) {
        fprintf(stdout, "Usage: iftPlateRecognition <train_orig_folder> <train_label_folder> <test_orig_folder> <test_label_folder> <output_folder> <output_metrics_file>\n");
        fprintf(stdout, "       train input images folder:      path to original grayscale images\n");
        fprintf(stdout, "       train label images folder:      path to label images\n");
        fprintf(stdout, "       test input images folder:       path to original grayscale images\n");
        fprintf(stdout, "       test label images folder:       path to label grayscale images\n");
        fprintf(stdout, "       output folder:                  path to output folder\n");
        fprintf(stdout, "       output metrics file:            path to metrics file\n");
        exit(1);
    }

    iftDir* trainOrigDir = iftLoadFilesFromDirBySuffix(argv[1], "pgm");
    iftDir* trainLabelDir = iftLoadFilesFromDirBySuffix(argv[2], "pgm");
    iftDir* testOrigDir = iftLoadFilesFromDirBySuffix(argv[3], "pgm");
    iftDir* testLabelDir = iftLoadFilesFromDirBySuffix(argv[4], "pgm");
    samplesPerImage = 15;

    Z = iftCreateDataSet(samplesPerImage*(trainOrigDir->nfiles), (DESC_FREQ_X + DESC_FREQ_Y) );
    Z->nclasses = 2;
    Z->ntrainsamples = 0;
    // Train samples
    indexZ = 0;
    totalInterCoef = 0;
    totalInterGT = 0;
    tp = 0;
    fp = 0;
    fn = 0;
    for (int i = 0; i < trainOrigDir->nfiles; ++i) {

        trainOrig = iftReadImageByExt(trainOrigDir->files[i]->pathname);
        trainLabel = iftReadImageByExt(trainLabelDir->files[i]->pathname);

        gradientImg = enhanceImage(trainOrig);

        compX = countComponentsX(gradientImg);
        compY = countComponentsY(gradientImg);
        selectTrainSamples(trainOrig, trainLabel, gradientImg, samplesPerImage, compX, compY, Z, indexZ);

        indexZ += samplesPerImage;

        iftDestroyImage(&trainOrig);
        iftDestroyImage(&gradientImg);
        iftDestroyImage(&compX);
        iftDestroyImage(&compY);
    }
    /* Normalize dataset */
    Zn = iftNormalizeDataSetByGlobalMax(Z);
    iftDestroyDataSet(&Z);
    iftSetStatus(Zn, IFT_TRAIN);

    // Train Classifier
    iftSVM* svm = iftCreateRBFSVC(1e4, 0.1);
    iftSVMTrainOVA(svm, Zn);

    time = 0;
    timeOCR = 0;

    // Apply classifier
    accCharacter = 0;
    meanNumCompChar = 0;
    for (int i = 0; i < testOrigDir->nfiles; ++i) {
        testOrig = iftReadImageByExt(testOrigDir->files[i]->pathname);
        testLabel = iftReadImageByExt(testLabelDir->files[i]->pathname);


        t1 = iftTic();
        gradientImg = enhanceImage(testOrig);
        sprintf(outFileTest, "%s%s", argv[5], iftFilename(testOrigDir->files[i]->pathname, NULL));
        // Detect plates
        testResult = detectPlate(testOrig, testLabel, svm, outFileTest);

        filtResult = filteringCandidates(testOrig, testResult, testLabel, outFileTest);
        t2 = iftToc();

        sprintf(outfilePlate, "%s%s.plate.pgm", argv[5], iftFilename(testOrigDir->files[i]->pathname, NULL));
        printf("%s\n", outfilePlate);

        plate = correctPlate(testOrig, filtResult);
        if (WRITE_TEMP_IMGS)
            iftWriteImageP2(plate, outfilePlate);

        iftImage* characters = iftAwesomeCharacterSeparation(plate);
        iftWriteImageP2(characters, "%s_characters.pgm", outFileTest);
        time += iftCompTime(t1, t2);

        // Character recognition
        t1 = iftTic();
        opticalCharacterRecognition(characters, outFileTest);
        t2 = iftToc();
        timeOCR += iftCompTime(t1, t2);

        computeMetrics(filtResult, testLabel, &tp, &fp, &fn, &intersecCoef, &intersecGT);
        totalInterCoef += intersecCoef;
        totalInterGT += intersecGT;

        iftDestroyImage(&gradientImg);
        iftDestroyImage(&testOrig);
        iftDestroyImage(&testLabel);
        iftDestroyImage(&testResult);
        iftDestroyImage(&filtResult);
        iftDestroyImage(&plate);
        iftDestroyImage(&characters);

    }

    precision = ((float)tp)/((float)(tp+fp));
    recall = ((float)tp)/((float)(tp+fn));
    totalInterCoef = totalInterCoef / ((float) testOrigDir->nfiles);
    totalInterGT = totalInterGT / ((float) testOrigDir->nfiles);
    time/= (float)testOrigDir->nfiles;
    timeOCR/= (float)testOrigDir->nfiles;
    accCharacter /= (float)testOrigDir->nfiles;
    meanNumCompChar /= (float)testOrigDir->nfiles;

    printf("Precision: %f\n", precision);
    printf("Recall: %f\n", recall);
    printf("Intersection Coef: %f\n", totalInterCoef);
    printf("Intersection GT: %f\n", totalInterGT);
    printf("Mean Detection time: %f\n", time);
    printf("Mean OCR time: %f\n", timeOCR);


    // Write file with results
    f1 = fopen(argv[6], "w");
    fprintf(f1,"%f\n", precision);
    fprintf(f1,"%f\n", recall);
    fprintf(f1,"%f\n", totalInterCoef);
    fprintf(f1,"%f\n", totalInterGT);
    fprintf(f1,"%d\n", tp);
    fprintf(f1,"%d\n", fp);
    fprintf(f1,"%d\n", fn);
    fprintf(f1, "%f\n", time);
    fclose(f1);

    iftDestroyDir(&trainOrigDir);
    iftDestroyDir(&trainLabelDir);
    iftDestroyDir(&testOrigDir);
    iftDestroyDir(&testLabelDir);


    return (0);
}

iftImage* iftAwesomeCharacterSeparation(iftImage* correctedPlate) {
    int numComp;
    int t = iftOtsu(correctedPlate);
    iftImage *relabel;
    int *arrNumPix, *arrXmax, *arrXmin, *arrYmax, *arrYmin;
    float ratio;

    // Compute thresholded image
    iftAdjRel* adj = iftRectangular(2.0, 4.0);
    iftImage* thresh = iftThreshold(correctedPlate, 0.75*t, iftMaximumValue(correctedPlate), 1);

    for (int p = 0; p < thresh->n; ++p) {
        if (thresh->val[p] == 0) {
            thresh->val[p] = 1;
        }
        else {
            thresh->val[p] = 0;
        }
    }

    iftImage* labels = iftLabelComp(thresh, adj);

    // Filter characters by size
    numComp = iftMaximumValue(labels);
    // Initialize arrays
    arrNumPix = iftAllocIntArray(numComp);
    arrXmax = iftAllocIntArray(numComp);
    arrXmin = iftAllocIntArray(numComp);
    arrYmax = iftAllocIntArray(numComp);
    arrYmin = iftAllocIntArray(numComp);
    for (int j = 0; j < numComp; ++j) {
        arrXmax[j] = -1;
        arrXmin[j] = INFINITY_INT;
        arrYmax[j] = -1;
        arrYmin[j] = INFINITY_INT;
    }

    for (int p = 0; p < labels->n; p++) {
        if(labels->val[p]>0) {
            int compLabel = labels->val[p] - 1;
            arrNumPix[compLabel]++;
            iftVoxel u = iftGetVoxelCoord(labels, p);
            if (u.x > arrXmax[compLabel]) {
                arrXmax[compLabel] = u.x;
            } else if (u.x < arrXmin[compLabel]) {
                arrXmin[compLabel] = u.x;
            }
            if (u.y > arrYmax[compLabel]) {
                arrYmax[compLabel] = u.y;
            } else if (u.y < arrYmin[compLabel]) {
                arrYmin[compLabel] = u.y;
            }
        }

    }

    for (int p = 0; p < labels->n; p++) {
        int compLabel = labels->val[p] - 1;
        int compXsize = arrXmax[compLabel] - arrXmin[compLabel];
        int compYsize = arrYmax[compLabel] - arrYmin[compLabel];
        ratio = (float) compYsize / (float) compXsize;
        if (arrNumPix[compLabel] < MIN_NUM_PIX_CHAR || ratio < RATIO_CHAR || compYsize<MIN_Y_CHAR_SIZE || compYsize>MAX_Y_CHAR_SIZE) {
            // Comment the next line to disable shape character filtering
            labels->val[p] = 0;
        }
    }

    relabel = iftLabelComp(labels, adj);

    iftDestroyAdjRel(&adj);
    iftDestroyImage(&thresh);
    iftDestroyImage(&labels);
    free(arrNumPix);

    return relabel;
}

void opticalCharacterRecognition(iftImage* img, const char* filename) {
    char cmd[200];

    for (int j = 0; j < img->n; ++j) {
        img->val[j] = img->val[j]? 0 : 255;
    }

    sprintf(cmd, "tesseract tmp.pgm %s_char", filename);

    iftWriteImageP2(img, "tmp.pgm");
    //system("convert tmp.pgm -resize 500x200 tmp.pgm");
    system(cmd);
}

void singularOpticalCharacterRecognition(iftImage* img, const char* filename) {
    char cmd[200];

    iftImage* tmpImg = iftCopyImage(img);

    int ncandidates = iftMaximumValue(img);


    for (int c = 1; c <= ncandidates; ++c) {
        for (int j = 0; j < img->n; ++j) {
            tmpImg->val[j] = img->val[j] == c? 0 : 255;
        }
        iftWriteImageP2(tmpImg, "tmp.pgm");
        sprintf(cmd, "tesseract tmp.pgm tmp_%d -psm 10", c);
        system(cmd);
    }

    sprintf(cmd, "paste -d\'\\0\' tmp_*.txt > %s.txt", filename);
    system(cmd);
    system("rm tmp_*.txt");
}


iftImage* iftPlateCharacterSeparation(iftImage* correctedPlate) {
    iftImage   *plate = iftEqualize(correctedPlate, 255);
    int mean=0;
    for (int p=0; p < plate->n; p++) {
        mean += plate->val[p];
    }
    mean /= plate->n;

    iftImage   *vclose  = iftVolumeClose(plate,5000);
    iftAdjRel  *A       = iftCircular(1.5);
    iftImage   *label   = iftWaterGray(plate,vclose,A);


    iftDestroyImage(&plate);
    iftDestroyImage(&vclose);
    //iftDestroyImage(&label);
    iftDestroyAdjRel(&A);

    return label;
}

void computeMetrics(iftImage *testResult, iftImage *testLabel, int *tp, int *fp, int *fn, float *intersecCoef, float *intersecGT) {
    int p, fneg, areaGT, ncomp;
    int *areaInter, *areaComp;
    iftAdjRel *A;
    float intersectionCoef, intersectionGT;
    iftImage *candResult;
    A = iftCircular(1.5);
    fneg = 1;
    // Count number of pixels from GT
    areaGT = 0;
    for(p = 0; p < testLabel->n; p++){
        if (testLabel->val[p] != 0) {
            areaGT++;
        }
    }
    // Relabel testResult to enumerate connected components
    candResult = iftRelabelRegions(testResult, A);
    // Compute areas
    ncomp = iftMaximumValue(candResult);
    areaComp = iftAllocIntArray(ncomp);
    areaInter = iftAllocIntArray(ncomp);
    for(p = 0; p < testLabel->n; p++){
        if (candResult->val[p] > 0) {
            areaComp[candResult->val[p]-1]++;
            if (testLabel->val[p] != 0) {
                areaInter[candResult->val[p]-1]++;
            }
        }
    }
    *intersecCoef = 0;
    *intersecGT = 0;
    for (int i = 0; i < ncomp; ++i) {
        intersectionCoef = ((float)areaInter[i]) / ((float) (areaComp[i] + areaGT - areaInter[i]));
        intersectionGT = ((float)areaInter[i]) / ((float) areaGT);
        if (intersectionCoef > *intersecCoef)
            *intersecCoef = intersectionCoef;
        if (intersectionGT > *intersecGT)
            *intersecGT = intersectionGT;

        if (intersectionGT >= 0.5 && intersectionCoef >= 0.3) {
            *tp += 1;
            fneg = 0;
        } else {
            *fp += 1;
        }
    }
    *fn += fneg;
    printf("IntCoef : %f, IntGT: %f, TP: %d, FP: %d, FN: %d\n", *intersecCoef, *intersecGT, *tp, *fp, *fn);

    free(areaComp);
    free(areaInter);
    iftDestroyAdjRel(&A);
    iftDestroyImage(&candResult);
}

iftImage *getComponentBasedMarkers(iftImage *relabelThrImg, char *filename) {
    iftImage *markers, *markersObj, *markersBg, *markersBgErode, *markersObjDil;
    int ncompThr;
    int *arrNumPix, *arrXmax, *arrXmin, *arrYmax, *arrYmin, *arrBoundary, *arrOutsideComp, *arrIsCharacter;
    float ratio;
    int compXsize, compYsize;
    char relthrFilename[200];
    markers = iftCreateImage(relabelThrImg->xsize, relabelThrImg->ysize, relabelThrImg->zsize);
    markersObj = iftCreateImage(relabelThrImg->xsize, relabelThrImg->ysize, relabelThrImg->zsize);
    markersBg = iftCreateImage(relabelThrImg->xsize, relabelThrImg->ysize, relabelThrImg->zsize);

    ncompThr = iftMaximumValue(relabelThrImg);

    // Initialize arrays
    arrNumPix = iftAllocIntArray(ncompThr);
    arrXmax = iftAllocIntArray(ncompThr);
    arrXmin = iftAllocIntArray(ncompThr);
    arrYmax = iftAllocIntArray(ncompThr);
    arrYmin = iftAllocIntArray(ncompThr);
    arrBoundary = iftAllocIntArray(ncompThr);
    arrOutsideComp = iftAllocIntArray(ncompThr);
    arrIsCharacter = iftAllocIntArray(ncompThr);
    for (int j = 0; j < ncompThr; ++j) {
        arrXmax[j] = -1;
        arrXmin[j] = INFINITY_INT;
        arrYmax[j] = -1;
        arrYmin[j] = INFINITY_INT;
    }
    // Initialize markers
    for (int p = 0; p < markers->n; ++p) {
        markers->val[p] = -1;
    }
    // Count number of pixel, xsize and ysize of every component
    for (int p = 0; p < relabelThrImg->n; p++) {
        if(relabelThrImg->val[p]>0) {
            int compLabel = relabelThrImg->val[p] - 1;
            arrNumPix[compLabel]++;
            iftVoxel u = iftGetVoxelCoord(relabelThrImg, p);
            if (u.x > arrXmax[compLabel]) {
                arrXmax[compLabel] = u.x;
            } else if (u.x < arrXmin[compLabel]) {
                arrXmin[compLabel] = u.x;
            }
            if (u.y > arrYmax[compLabel]) {
                arrYmax[compLabel] = u.y;
            } else if (u.y < arrYmin[compLabel]) {
                arrYmin[compLabel] = u.y;
            }
            if (u.x == 0 || (u.x == (relabelThrImg->xsize-1)) || u.y == 0 || (u.y == (relabelThrImg->ysize-1))) {
                arrBoundary[compLabel] = 1;
            }
        }
    }
    // Filtering components by shape

    for (int j = 0; j < ncompThr; ++j) {
        compXsize = arrXmax[j] - arrXmin[j];
        compYsize = arrYmax[j] - arrYmin[j];
        ratio = (float) compYsize / (float) compXsize;
        int npix = arrNumPix[j];
        if (ratio > RATIO_CHAR && compYsize>MIN_Y_CHAR_SIZE && compYsize<MAX_Y_CHAR_SIZE && (npix > MIN_NUM_PIX_CHAR && npix < MAX_NUM_PIX_CHAR) && arrBoundary[j] ==0) {
            arrIsCharacter[j] = 1;
        } else if (arrBoundary[j]==1 && npix>180 && npix<2200) {
            arrOutsideComp[j] = 1;
        }
    }
    for (int p = 0; p < relabelThrImg->n; p++) {
        if(relabelThrImg->val[p]>0) {
            int compLabel = relabelThrImg->val[p] - 1;

            if (arrIsCharacter[compLabel] == 0) {
                relabelThrImg->val[p] = 0;
            } else {
                // Object seed
                markersObj->val[p] = 1;
            }
            if (arrOutsideComp[compLabel] == 1) {
                // Background seed
                relabelThrImg->val[p] = 40;
                markersBg->val[p] = 1;
            }
        }
    }
    sprintf(relthrFilename, "%s.relbl.pgm", filename);
    if (WRITE_TEMP_IMGS)
        iftWriteImageP2(relabelThrImg, relthrFilename);

    // Erode markersBg
    iftAdjRel *Aerode, *Adil;
    Aerode = iftCircular(2.0);
    markersBgErode = iftErode(markersBg, Aerode);
    iftDestroyAdjRel(&Aerode);

    Adil = iftRectangular(7.0, 2.0);
    markersObjDil = iftDilate(markersObj, Adil);
    iftDestroyAdjRel(&Adil);


    // Copy markersObj and markersBgErode
    for (int p = 0; p <markers->n ; p++) {
        if (markersBgErode->val[p]!=0) {
            markers->val[p] = 0;
        }
        if (markersObjDil->val[p]!=0) {
            markers->val[p] = 1;
        }
    }

    free(arrNumPix);
    free(arrXmax);
    free(arrXmin);
    free(arrYmax);
    free(arrYmin);
    free(arrBoundary);
    free(arrOutsideComp);
    free(arrIsCharacter);
    iftDestroyImage(&markersObj);
    iftDestroyImage(&markersBg);
    iftDestroyImage(&markersBgErode);
    iftDestroyImage(&markersObjDil);


    return markers;

}

iftImage *filteringCandidates(iftImage *orig, iftImage *candidates, iftImage *label, char *filename) {
    iftImage *filtImg, *relabelImg, *compImg, *compErode, *compDil, *markers, *basins, *sgmImg, *showMarker, *compMarkers;
    iftLabeledSet *seed;
    iftAdjRel *A, *Aerode, *Adil;
    char sgmFilename[200], labelFilename[200];
    int ncomp;
    A = iftRectangular(2.0, 4.0);
    Aerode = iftRectangular(54, 28);
    Adil = iftRectangular(30.0, 10.0);
    // Create seeds from characters
    iftVoxel uPlaca, vPlaca;
    iftImage *bbImg, *thrImg, *relabelThrImg, *invThrImg;
    int maxVal = iftMaximumValue(candidates);
    char thrFilename[200];

    getBoundingBoxCoord(candidates, maxVal, &uPlaca, &vPlaca);
    bbImg = createBoundingBox2DByCoord(orig, uPlaca, vPlaca);
    int t = iftOtsu(bbImg);

    t*=0.7;
    invThrImg = iftThreshold(bbImg, t, maxVal, 1);
    thrImg = iftCreateImage(bbImg->xsize, bbImg->ysize, bbImg->zsize);
    for (int p = 0; p < invThrImg->n; p++) {
        if (invThrImg->val[p] !=0) {
            thrImg->val[p] = 0;
        } else {
            thrImg->val[p] = 1;
        }
    }
    sprintf(thrFilename, "%s.thr_%d.pgm", filename, t);
    if (WRITE_TEMP_IMGS)
        iftWriteImageP2(thrImg , thrFilename);
    sprintf(labelFilename, "%s.gt_%d.pgm", filename, t);
    if (WRITE_TEMP_IMGS)
        iftWriteImageP2(label , labelFilename);
    // Relabel thresholded image
    relabelThrImg = iftLabelComp(thrImg, A);

    // Compute component-based internal and external markers
    compMarkers = getComponentBasedMarkers(relabelThrImg, filename);

    iftDestroyImage(&bbImg);
    iftDestroyImage(&thrImg);
    iftDestroyImage(&invThrImg);

    // compute gradient image
    basins = iftImageBasins(orig,A);
    iftImage* normBasins = iftNormalize(basins, 0, 255);
    if (WRITE_TEMP_IMGS)
        iftWriteImageP2(normBasins, "%s_grad.pgm", filename);
    iftDestroyImage(&normBasins);

    relabelImg = iftRelabelRegions(candidates, A);
    ncomp = iftMaximumValue(relabelImg);
    if (WRITE_TEMP_IMGS)
        iftWriteImageP2(candidates, "%s_cand.pgm", filename);
    filtImg = iftCreateImage(candidates->xsize, candidates->ysize, candidates->zsize);
    for (int i = 0; i < ncomp; i++) {
        compImg = iftCreateImage(candidates->xsize, candidates->ysize, candidates->zsize);
        for (int p = 0; p < candidates->n; p++) {
            if (relabelImg->val[p] == (i+1))
                compImg->val[p] = i+1;
        }
        compErode = iftErode(compImg, Aerode);
        compDil = iftDilate(compImg, Adil);
        markers = drawGrayMarkers(compDil, ncomp+1, compErode, A);
        // Merge compMarkers and markers

        for (int p = 0; p < compMarkers->n; p++) {
            if (compMarkers->val[p] >= 0) {
                int origp;
                iftVoxel u = iftGetVoxelCoord(compMarkers, p);
                u.x += uPlaca.x;
                u.y += uPlaca.y;
                origp = iftGetVoxelIndex(markers, u);
                markers->val[origp] = compMarkers->val[p];
            }
        }

        // Write showMarkers
        showMarker = iftCopyImage(markers);
        for (int p = 0; p < markers->n; p++) {
            showMarker->val[p]++;
        }
        seed = iftLabeledSetFromSeedImage(markers);
        sgmImg = iftWatershed(basins, A, seed);

        if (WRITE_TEMP_IMGS)
            iftWriteImageP2(showMarker , "%s.marker_%d.pgm", filename, (i+1));

        for (int p = 0; p < sgmImg->n; p++) {
            if (sgmImg->val[p]!=0) {
                filtImg->val[p] = sgmImg->val[p];
            }
        }

        sprintf(sgmFilename, "%s.sgm_%d.pgm", filename, (i+1));
        if (WRITE_TEMP_IMGS)
            iftWriteImageP2(filtImg , sgmFilename);

        iftDestroyImage(&compImg);
        iftDestroyImage(&compErode);
        iftDestroyImage(&compDil);
        iftDestroyImage(&showMarker);
        iftDestroyImage(&sgmImg);
    }


    iftAdjRel* rel = iftRectangular(40.0, 10.0);

    iftImage* open = iftOpen(filtImg, rel);
    if (WRITE_TEMP_IMGS)
        iftWriteImageP2(open, "%s_closed.pgm", filename);

    iftDestroyAdjRel(&rel);
    iftDestroyImage(&filtImg);
    iftDestroyImage(&relabelThrImg);
    iftDestroyImage(&compMarkers);
    iftDestroyAdjRel(&Aerode);
    iftDestroyAdjRel(&Adil);
    iftDestroyAdjRel(&A);
    iftDestroyImage(&relabelImg);
    iftDestroyImage(&basins);

    return open;
}

iftImage *drawGrayMarkers(iftImage *label, int bgMarkerValue, iftImage *objMarker, iftAdjRel *A)
{
    iftImage *img;
    iftVoxel u,v;
    int i,p,q;
    img = iftCopyImage(objMarker);
    for (p = 0; p < img->n ; p++) {
        if (img->val[p] == 0) {
            img->val[p] = - 1;
        }
    }
    for (p=0; p < img->n; p++) {
        u.x = iftGetXCoord(label,p);
        u.y = iftGetYCoord(label,p);
        u.z = iftGetZCoord(label,p);
        for (i=0; i < A->n; i++) {
            v.x = u.x + A->dx[i];
            v.y = u.y + A->dy[i];
            v.z = u.z + A->dz[i];
            if (iftValidVoxel(label,v)){
                q = iftGetVoxelIndex(label,v);
                if (label->val[p] < label->val[q]){
                    img->val[p] = 0;
                    break;
                }
            }
        }
    }


    return img;
}

//Features: lbp bins (256)
iftFeatures* computeLBP(iftImage *img) {
    int i, p;
    int lbp_code;
    iftImage *lbpImg;
    iftAdjRel *A;
    A = iftCircular(sqrt(2.0));
    iftFeatures *lbp = iftCreateFeatures(256);
    lbpImg = iftCreateImage(img->xsize, img->ysize, img->zsize);

    for(p = 0; p < img->n; p++){
        lbp_code = 0;
        iftVoxel u = iftGetVoxelCoord(img, p);

        for(i = 1; i < A->n; i++){
            lbp_code = lbp_code << 1;

            iftVoxel v = iftGetAdjacentVoxel(A, u, i);

            if(iftValidVoxel(img, v)){
                int q = iftGetVoxelIndex(img, v);

                if(img->val[p] > img->val[q])
                    lbp_code = lbp_code + 1;
            }
        }
        lbp->val[lbp_code] = lbp->val[lbp_code] + 1;
        lbpImg->val[p] = lbp_code;
    }
    for(i = 0; i < 256; i++){
        lbp->val[i] = lbp->val[i] / (float)img->n;
    }

    iftDestroyAdjRel(&A);
    return lbp;
}

iftImage* selectStongestCandidate(iftImage* img, const char* filename) {


    // Apply otsu threshold
    int t = iftOtsu(img);


    iftImage *thrImg = iftThreshold(img, t, 255, 255);


    iftAdjRel* adjRel = iftCircular(2.0);
    iftImage* labeledImage = iftFastLabelComp(thrImg, adjRel);

    int ncandidates = iftMaximumValue(labeledImage);

    if(ncandidates>1) {

        float *candidateScore = iftAllocFloatArray(ncandidates + 1);
        int *candidatesNumber = iftAllocIntArray(ncandidates + 1);

        for (int i = 0; i < labeledImage->n; ++i) {
            int c = labeledImage->val[i];
            candidateScore[c] += img->val[i];
            candidatesNumber[c]++;
        }

        int imax = 0;

        for (int i = 0; i <= ncandidates; ++i) {
            if (candidatesNumber[i] != 0) {
                candidateScore[i] /= candidatesNumber[i];
            }
            if (candidateScore[imax] < candidateScore[i]) {
                imax = i;
            }
        }

        for (int p = 0; p < labeledImage->n; ++p) {
            if (labeledImage->val[p] == imax) {
                thrImg->val[p] = 255;
            }
            else {
                thrImg->val[p] = 0;
            }
        }
        free(candidateScore);
        free(candidatesNumber);
    }

    free(labeledImage);

    return thrImg;
}

iftImage *detectPlate(iftImage *orig, iftImage *label, iftSVM* svm, char *filename) {
    iftDataSet *Z;
    iftFeatures *feat;
    iftImage *gradientImg, *compX, *compY, *normCandImg;
    iftFImage *candImg;
    iftVoxel u,v;
    int i, j, strideX, strideY, numCand, nsamples, indexZ;
    strideX = 3;
    strideY = 3;
    u.z = 0;
    v.z = 0;
    numCand = 0;

    /* Extract features */
    gradientImg = enhanceImage(orig);
    compX = countComponentsX(gradientImg);
    compY = countComponentsY(gradientImg);

    nsamples = ((orig->xsize - BOX_SIZE_X)/strideX) * ((orig->ysize - BOX_SIZE_Y)/strideY);

    /* Create dataset  */
    Z = iftCreateDataSet(nsamples, (DESC_FREQ_X + DESC_FREQ_Y));
    //Z = iftCreateDataSet(nsamples, 256);
    iftSetStatus(Z, IFT_TRAIN);
    candImg = iftCreateFImage(orig->xsize, orig->ysize, orig->zsize);
    indexZ = 0;
    for (j = 0; j < (orig->xsize - BOX_SIZE_X - strideX + 1); j+=strideX) {
        for (i = 0; i < (orig->ysize - BOX_SIZE_Y - strideY + 1); i += strideY) {
            u.x = j;
            u.y = i;
            v.x = u.x + BOX_SIZE_X - 1;
            v.y = u.y + BOX_SIZE_Y - 1;
            feat = getFreqFeatures(compX, compY, u, v);


            // if is a candidate sample (number of changes greater than MIN_CH)
            if (sumFeatureValues(feat) > MIN_CH) {
                // copy feature values
                for (int k = 0; k < feat->n; k++) {
                    Z->sample[indexZ].feat[k] = (feat->val[k] / (float)NORM_VAL_FEAT_CH);
                    Z->sample[indexZ].status = IFT_TEST;
                    Z->sample[indexZ].id = indexZ + 1;
                }

            }
            iftDestroyFeatures(&feat);
            indexZ++;
        }
    }

    /* Classify selected samples */
    iftSVMClassifyOVA(svm, Z, IFT_TEST);

    /* Write normalized candidate image */
    indexZ = 0;
    for (j = 0; j < orig->xsize - BOX_SIZE_X; j+=strideX) {
        for (i = 0; i < orig->ysize - BOX_SIZE_Y; i += strideY) {
            u.x = j;
            u.y = i;
            v.x = u.x + BOX_SIZE_X - 1;
            v.y = u.y + BOX_SIZE_Y - 1;

            if ( Z->sample[indexZ].label == 1) {
                addValueToBoundingBox(candImg, u, v, Z->sample[indexZ].weight);
                numCand++;
            }
            indexZ++;
        }
    }

    normCandImg = iftFImageToImage(candImg, 255);

    iftImage* thrImg = selectStongestCandidate(normCandImg, filename);

    iftDestroyImage(&normCandImg);
    iftDestroyFImage(&candImg);
    iftDestroyImage(&gradientImg);
    iftDestroyImage(&compX);
    iftDestroyImage(&compY);

    return thrImg;
}

iftImage *countComponentsX(iftImage *orig) {
    int p, lastElem, countX;
    iftImage *compX;
    compX = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
    for (int j = 0; j < orig->xsize; j++) {
        lastElem = -1;
        countX = 0;
        for (int i = 0; i < orig->ysize; i++) {
            p = j + i*orig->xsize;
            if (orig->val[p] != lastElem) {
                countX++;
            }
            compX->val[p] = countX;
            lastElem = orig->val[p];
        }
    }
    return compX;
}

iftImage *countComponentsY(iftImage *orig) {
    int p, lastElem, countY;
    iftImage *compY;
    compY = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
    for (int i = 0; i < orig->ysize; i++) {
        lastElem = -1;
        countY = 0;
        for (int j = 0; j < orig->xsize; j++) {
            p = j + i*orig->xsize;
            if (orig->val[p] != lastElem) {
                countY++;
            }
            compY->val[p] = countY;
            lastElem = orig->val[p];
        }
    }
    return compY;
}

float sumFeatureValues(iftFeatures *feat) {
    float count = 0;
    for (int i = 0; i < feat->n; ++i) {
        count += feat->val[i];
    }
    return count;
}

iftFeatures *getFreqFeatures(iftImage *compX, iftImage *compY, iftVoxel u, iftVoxel v) {

    /* u: voxel top left, v: voxel bottom right */

    int p, q, restX, restY, indexX, indexY, bsizeX, real_bsizeX, bsizeY, real_bsizeY;
    int countX, countY, totalX, totalY;
    iftFeatures *feat = iftCreateFeatures(DESC_FREQ_X + DESC_FREQ_Y);
    bsizeX = (v.x - u.x + 1) / DESC_FREQ_X;
    restX = (v.x - u.x + 1) % DESC_FREQ_X;
    bsizeY = (v.y - u.y + 1) / DESC_FREQ_Y;
    restY = (v.y - u.y + 1) % DESC_FREQ_Y;
    indexX = 0;
    countX = 0;
    totalX = 0;
    totalY = 0;
    real_bsizeX = bsizeX;
    real_bsizeY = bsizeY;
    // Count changes in X
    for (int j = u.x; j <= v.x; j++) {
        p =  j + u.y * compX->xsize; // top
        q =  j + v.y * compX->xsize; // bottom
        feat->val[indexX] += (float)(compX->val[q] - compX->val[p]);
        totalX += (compX->val[q] - compX->val[p]);

        // compute bucket
        if (countX ==0) {
            real_bsizeX = bsizeX;
            if (restX > 0) {
                real_bsizeX = bsizeX + 1;
                restX--;
            }
        }
        countX += 1;
        if (countX >= real_bsizeX) {
            countX = 0;
            indexX++;
        }
    }

    indexY = 0;
    countY = 0;
    // Count changes in Y
    for (int i = u.y; i <= v.y; i++) {
        p =  u.x + i * compY->xsize; // left
        q =  v.x + i * compY->xsize; // right
        feat->val[DESC_FREQ_X + indexY] += (float)(compY->val[q] - compY->val[p]);
        totalY += (compY->val[q] - compY->val[p]);

        // compute bucket
        if (countY ==0) {
            real_bsizeY = bsizeY;
            if (restY > 0) {
                real_bsizeY = bsizeY + 1;
                restY--;
            }
        }
        countY += 1;
        if (countY >= real_bsizeY) {
            countY = 0;
            indexY++;
        }
    }

    return feat;
}

void getBoundingBoxCoord(iftImage *label, int val, iftVoxel *u, iftVoxel *v) {
    iftVoxel tmp, min, max;
    int p;
    min.x = INFINITY_INT;
    min.y = INFINITY_INT;
    min.z = 0;
    max.x = -1;
    max.y = -1;
    max.z = 0;
    // Find bounding box
    for (p=0; p < label->n; p++) {
        if (label->val[p] == val) {
            tmp = iftGetVoxelCoord(label,p);
            if (tmp.x < min.x)
                min.x = tmp.x;
            else if (tmp.x > max.x)
                max.x = tmp.x;

            if (tmp.y < min.y)
                min.y = tmp.y;
            else if (tmp.y > max.y)
                max.y = tmp.y;
        }
    }
    *u = min;
    *v = max;
}

void selectTrainSamples(iftImage *orig, iftImage *label, iftImage *gradientImg, int nsamples, iftImage *compX, iftImage *compY, iftDataSet *Z, int indexZ) {
    iftVoxel uPlaca, vPlaca, u, v;
    iftFeatures *feat;
    iftImage *negSampleImg;
    negSampleImg = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
    // Select positive samples
    getBoundingBoxCoord(label, 1, &uPlaca, &vPlaca);
    feat = getFreqFeatures(compX, compY, uPlaca, vPlaca);
    // Populate dataset with nunnormalized features
    for (int i = 0; i < feat->n; ++i) {
        Z->sample[indexZ].feat[i] = feat->val[i];
        Z->sample[indexZ].truelabel = 1;
    }
    indexZ++;
    setBoundingBox(label, uPlaca, vPlaca);
    nsamples--;
    iftDestroyFeatures(&feat);

    /* Select general negative samples */
    while (nsamples > 0) {
        u.z = 0;
        v.z = 0;
        u.x = randInterval(0, orig->xsize - BOX_SIZE_X);
        u.y = randInterval(0, orig->ysize - BOX_SIZE_Y);
        v.x = u.x + BOX_SIZE_X - 1;
        v.y = u.y + BOX_SIZE_Y - 1;
        if (isAvailableVoxel(label, u, v)) {
            if (isCandidateSample(compX, compY, u, v)) {
                feat = getFreqFeatures(compX, compY, u, v);
                // Populate dataset with nunnormalized features
                for (int i = 0; i < feat->n; ++i) {
                    Z->sample[indexZ].feat[i] = feat->val[i];
                    Z->sample[indexZ].truelabel = 2;
                }
                indexZ++;
                setBoundingBox(negSampleImg, u, v);
                nsamples--;
                iftDestroyFeatures(&feat);
            }
        }
    }

    iftDestroyImage(&negSampleImg);
}

int isAvailableVoxel(iftImage *label, iftVoxel u, iftVoxel v) {
    int isAvalable, p;
    iftVoxel tmp;
    isAvalable = 1;

    if ( !iftValidVoxel(label, u) || !iftValidVoxel(label, v)) {
        return 0;
    }
    p = iftGetVoxelIndex(label, u);
    // check top left
    if (label->val[p] > 0)
        isAvalable = 0;
    // check top right
    tmp.z = 0;
    tmp.x = v.x;
    tmp.y = u.y;
    p = iftGetVoxelIndex(label, tmp);
    if (label->val[p] > 0)
        isAvalable = 0;
    // check middle point
    tmp.x = (u.x + v.x)/2;
    tmp.y = (u.y + v.y)/2;
    p = iftGetVoxelIndex(label, tmp);
    if (label->val[p] > 0)
        isAvalable = 0;
    // check bottom left
    tmp.x = u.x;
    tmp.y = v.y;
    p = iftGetVoxelIndex(label, tmp);
    if (label->val[p] > 0)
        isAvalable = 0;
    // check bottom right
    p = iftGetVoxelIndex(label, v);
    if (label->val[p] > 0)
        isAvalable = 0;
    return isAvalable;
}

int isCandidateSample(iftImage *compX, iftImage *compY, iftVoxel u, iftVoxel v) {

    /* u: voxel top left, v: voxel bottom right */
    int p, q;
    int totalX, totalY;

    totalX = 0;
    totalY = 0;

    for (int j = u.x; j <= v.x; j++) {
        p =  j + u.y * compX->xsize; // top
        q =  j + v.y * compX->xsize; // bottom
        totalX += (compX->val[q] - compX->val[p]);
    }

    for (int i = u.y; i <= v.y; i++) {
        p =  u.x + i * compY->xsize; // left
        q =  v.x + i * compY->xsize; // right
        totalY += (compY->val[q] - compY->val[p]);
    }

    if ((totalX + totalY) > MIN_CH)
        return 1;
    else
        return 0;
}

void setBoundingBox(iftImage *label, iftVoxel u, iftVoxel v) {
    /* write 1 in the label image for the bounding box delimited by u and v (voxels) */
    int p;
    for (int i = u.y; i <= v.y; i++) {
        for (int j = u.x; j <= v.x; j++) {
            p = j + i * label->xsize;
            label->val[p] = 1;
        }
    }
}

void addValueToBoundingBox(iftFImage *label, iftVoxel u, iftVoxel v, float value) {
    /* increment by 1 pixel values in the label image for the bounding box delimited by u and v (voxels) */
    int p;
    for (int i = u.y; i <= v.y; i++) {
        for (int j = u.x; j <= v.x; j++) {
            p = j + i * label->xsize;
            label->val[p] += value;
        }
    }
}

int randInterval(int min, int max) {
    int r;
      int range = 1 + max - min;
      int buckets = RAND_MAX / range;
      int limit = buckets * range;

    /* Create equal size buckets all in a row, then fire randomly towards
     * the buckets until you land in one of them. All buckets are equally
     * likely. If you land off the end of the line of buckets, try again. */
    do
    {
        r = rand();
    } while (r >= limit);

    return min + (r / buckets);
}

iftDataSet *iftNormalizeDataSetByGlobalMax(iftDataSet *Z)
{
    /* Normalize dataset using the global maximum */
    int s,i;
    float max;
    max = INFINITY_FLT_NEG;
    iftDataSet *Zn = iftCopyDataSet(Z, true);

    for (s=0; s < Zn->nsamples; s++) {
        for (i=0; i < Zn->nfeats; i++) {
            if (max < Zn->sample[s].feat[i]) {
                max = Zn->sample[s].feat[i];
            }
        }
    }
    max = NORM_VAL_FEAT_CH;

    if (max != 0) {
        for (s = 0; s < Zn->nsamples; s++) {
            for (i = 0; i < Zn->nfeats; i++) {
                Zn->sample[s].feat[i] = Zn->sample[s].feat[i] / max;
            }
        }
    }

    return(Zn);
}

iftDataSet *iftNormalizeDataSetByLocalMax(iftDataSet *Z)
{
    /* Normalize dataset using the local maximum */
    int s,i;
    float *maxArr;

    iftDataSet *Zn = iftCopyDataSet(Z);
    maxArr = iftAllocFloatArray(Zn->nsamples);
    for (i=0; i < Zn->nsamples; i++) {
        maxArr[i] = INFINITY_FLT_NEG;
    }

    for (s=0; s < Zn->nsamples; s++) {
        for (i=0; i < Zn->nfeats; i++) {
            if (maxArr[s] < Zn->sample[s].feat[i]) {
                maxArr[s] = Zn->sample[s].feat[i];
            }
        }
    }

    for (s = 0; s < Zn->nsamples; s++) {
        if (maxArr[s] != 0) {
            for (i = 0; i < Zn->nfeats; i++) {
                Zn->sample[s].feat[i] = Zn->sample[s].feat[i] / maxArr[s];
            }
        }
    }

    return(Zn);
}

iftImage *enhanceImage(iftImage *orig) {

    iftImage *aux[3];
    iftKernel *K = NULL;
    iftAdjRel *A = NULL;

    A = iftCircular(5.0);
    aux[0] = iftNormalizeImage(orig, A, 4095);

    aux[1] = iftCloseBasins(aux[0],NULL,NULL);
    aux[2] = iftSub(aux[1], aux[0]);
    //aux[2] = iftSub(aux[2], aux[0]);
    iftDestroyImage(&aux[1]);

    aux[0] = aux[2];

    int max = iftMaximumValue(aux[0]);
    int t = iftOtsu(aux[0]);

    aux[2] = iftThreshold(aux[0], t, max, 255);

    iftImage* final = iftCopyImage(aux[2]);

    iftDestroyImage(&aux[0]);
    iftDestroyImage(&aux[1]);
    iftDestroyImage(&aux[2]);

    iftDestroyAdjRel(&A);
    iftDestroyKernel(&K);

    return final;
}

iftImage *enhanceImageFuzzy(iftImage *orig) {

    iftImage *aux[3];
    iftKernel *K = NULL;
    iftAdjRel *A = NULL;

    A = iftCircular(5.0);
    aux[0] = iftNormalizeImage(orig, A, 4095);

    aux[1] = iftCloseBasins(aux[0],NULL,NULL);
    aux[2] = iftSub(aux[1], aux[0]);

    iftImage* final = iftCopyImage(aux[2]);

    iftDestroyImage(&aux[0]);
    iftDestroyImage(&aux[1]);
    iftDestroyImage(&aux[2]);

    iftDestroyAdjRel(&A);
    iftDestroyKernel(&K);

    return final;
}


iftImage *createBoundingBox2DByCoord(iftImage *img, iftVoxel u, iftVoxel v) {
    /* u: voxel top left, v: voxel bottom right */
    iftImage *out;
    int minX, minY, maxX, maxY;
    int p, i, j, origp;
    minX = u.x;
    minY = u.y;
    maxX = v.x;
    maxY = v.y;
    out = iftCreateImage((maxX-minX+1), (maxY-minY+1), img->zsize);

    // Alloc Cb and Cr channels for color images
    if (img->Cb != NULL) {
        out->Cb = iftAllocUShortArray(img->n);
        out->Cr = iftAllocUShortArray(img->n);
    }
    // Copy pixel values
    for (i=minY; i < (maxY+1); i++) {
        for (j=minX; j < (maxX+1); j++) {
            origp = i * img->xsize + j;
            p = (i-minY) * (maxX-minX+1) + (j-minX);
            out->val[p] = img->val[origp];
            // Copy Cb and Cr bands for color images
            if (img->Cb != NULL) {
                out->Cb[p] = img->Cb[origp];
                out->Cr[p] = img->Cr[origp];
            }
        }
    }

    return out;
}


iftMatrix *iftCameraCorrectionMatrix(iftPoint *src, iftPoint *dst)
{
    iftMatrix *A = iftCreateMatrix(8,8), *c = iftCreateMatrix(1,8);
    iftMatrix *invA, *b, *B;
    B = NULL;

    A->val[iftGetMatrixIndex(A,0,0)] = src[0].x;
    A->val[iftGetMatrixIndex(A,1,0)] = src[0].y;
    A->val[iftGetMatrixIndex(A,2,0)] = 1.0;
    A->val[iftGetMatrixIndex(A,6,0)] = -src[0].x*dst[0].x;
    A->val[iftGetMatrixIndex(A,7,0)] = -src[0].y*dst[0].x;

    A->val[iftGetMatrixIndex(A,3,1)] = src[0].x;
    A->val[iftGetMatrixIndex(A,4,1)] = src[0].y;
    A->val[iftGetMatrixIndex(A,5,1)] = 1.0;
    A->val[iftGetMatrixIndex(A,6,1)] = -src[0].x*dst[0].y;
    A->val[iftGetMatrixIndex(A,7,1)] = -src[0].y*dst[0].y;


    A->val[iftGetMatrixIndex(A,0,2)] = src[1].x;
    A->val[iftGetMatrixIndex(A,1,2)] = src[1].y;
    A->val[iftGetMatrixIndex(A,2,2)] = 1.0;
    A->val[iftGetMatrixIndex(A,6,2)] = -src[1].x*dst[1].x;
    A->val[iftGetMatrixIndex(A,7,2)] = -src[1].y*dst[1].x;

    A->val[iftGetMatrixIndex(A,3,3)] = src[1].x;
    A->val[iftGetMatrixIndex(A,4,3)] = src[1].y;
    A->val[iftGetMatrixIndex(A,5,3)] = 1.0;
    A->val[iftGetMatrixIndex(A,6,3)] = -src[1].x*dst[1].y;
    A->val[iftGetMatrixIndex(A,7,3)] = -src[1].y*dst[1].y;

    A->val[iftGetMatrixIndex(A,0,4)] = src[2].x;
    A->val[iftGetMatrixIndex(A,1,4)] = src[2].y;
    A->val[iftGetMatrixIndex(A,2,4)] = 1.0;
    A->val[iftGetMatrixIndex(A,6,4)] = -src[2].x*dst[2].x;
    A->val[iftGetMatrixIndex(A,7,4)] = -src[2].y*dst[2].x;

    A->val[iftGetMatrixIndex(A,3,5)] = src[2].x;
    A->val[iftGetMatrixIndex(A,4,5)] = src[2].y;
    A->val[iftGetMatrixIndex(A,5,5)] = 1.0;
    A->val[iftGetMatrixIndex(A,6,5)] = -src[2].x*dst[2].y;
    A->val[iftGetMatrixIndex(A,7,5)] = -src[2].y*dst[2].y;


    A->val[iftGetMatrixIndex(A,0,6)] = src[3].x;
    A->val[iftGetMatrixIndex(A,1,6)] = src[3].y;
    A->val[iftGetMatrixIndex(A,2,6)] = 1.0;
    A->val[iftGetMatrixIndex(A,6,6)] = -src[3].x*dst[3].x;
    A->val[iftGetMatrixIndex(A,7,6)] = -src[3].y*dst[3].x;

    A->val[iftGetMatrixIndex(A,3,7)] = src[3].x;
    A->val[iftGetMatrixIndex(A,4,7)] = src[3].y;
    A->val[iftGetMatrixIndex(A,5,7)] = 1.0;
    A->val[iftGetMatrixIndex(A,6,7)] = -src[3].x*dst[3].y;
    A->val[iftGetMatrixIndex(A,7,7)] = -src[3].y*dst[3].y;

    //iftPrintMatrix(A);

    c->val[iftGetMatrixIndex(c,0,0)] = dst[0].x;
    c->val[iftGetMatrixIndex(c,0,1)] = dst[0].y;
    c->val[iftGetMatrixIndex(c,0,2)] = dst[1].x;
    c->val[iftGetMatrixIndex(c,0,3)] = dst[1].y;
    c->val[iftGetMatrixIndex(c,0,4)] = dst[2].x;
    c->val[iftGetMatrixIndex(c,0,5)] = dst[2].y;
    c->val[iftGetMatrixIndex(c,0,6)] = dst[3].x;
    c->val[iftGetMatrixIndex(c,0,7)] = dst[3].y;

    invA = iftInvertMatrixAw(A);
    if (invA == NULL) {
        return B;
    }
    b    = iftMultMatrices(invA,c);
    B    = iftCreateMatrix(3,3);

    B->val[iftGetMatrixIndex(B,0,0)] = b->val[iftGetMatrixIndex(b,0,0)];
    B->val[iftGetMatrixIndex(B,1,0)] = b->val[iftGetMatrixIndex(b,0,1)];
    B->val[iftGetMatrixIndex(B,2,0)] = b->val[iftGetMatrixIndex(b,0,2)];
    B->val[iftGetMatrixIndex(B,0,1)] = b->val[iftGetMatrixIndex(b,0,3)];
    B->val[iftGetMatrixIndex(B,1,1)] = b->val[iftGetMatrixIndex(b,0,4)];
    B->val[iftGetMatrixIndex(B,2,1)] = b->val[iftGetMatrixIndex(b,0,5)];
    B->val[iftGetMatrixIndex(B,0,2)] = b->val[iftGetMatrixIndex(b,0,6)];
    B->val[iftGetMatrixIndex(B,1,2)] = b->val[iftGetMatrixIndex(b,0,7)];
    B->val[iftGetMatrixIndex(B,2,2)] = 1.0;

    iftDestroyMatrix(&A);
    iftDestroyMatrix(&c);
    iftDestroyMatrix(&b);
    iftDestroyMatrix(&invA);

    return(B);
}

iftImage *iftCorrectImage(iftImage *img, iftPoint *src_pt, iftPoint *dst_pt, int xsize, int ysize)
{
    iftImage  *cimg;
    iftMatrix *T,*InvT, *src, *dst;
    iftPoint P;
    iftVoxel v,u;
    int q;

    T = iftCameraCorrectionMatrix(src_pt, dst_pt);
    if (T == NULL) {
        return img;
    }

    //iftPrintMatrix(T);

    InvT = iftInvertMatrixAw(T);
    if (InvT == NULL) {
        return img;
    }

    cimg = iftCreateImage(xsize,ysize,1);
    src = iftCreateMatrix(1,3);
    dst=iftCreateMatrix(1,3);

    v.z = u.z = P.z = 0;
    for (v.y = 0; v.y < cimg->ysize; v.y++)
        for (v.x = 0; v.x < cimg->xsize; v.x++){
            q    = iftGetVoxelIndex(cimg,v);
            dst->val[iftGetMatrixIndex(dst,0,0)] = v.x;
            dst->val[iftGetMatrixIndex(dst,0,1)] = v.y;
            dst->val[iftGetMatrixIndex(dst,0,2)] = 1;
            src   = iftMultMatrices(InvT,dst);
            P.x   = src->val[iftGetMatrixIndex(src,0,0)]/src->val[iftGetMatrixIndex(src,0,2)];
            P.y   = src->val[iftGetMatrixIndex(src,0,1)]/src->val[iftGetMatrixIndex(src,0,2)];
            u.x   = ROUND(P.x);
            u.y   = ROUND(P.y);

            if (iftValidVoxel(img,u)){
                cimg->val[q] = iftImageValueAtPoint2D(img,P);
            }
        }

    iftDestroyMatrix(&InvT);
    iftDestroyMatrix(&src);
    iftDestroyMatrix(&dst);
    iftDestroyMatrix(&T);

    return(cimg);
}

iftPoint *iftTheFourPlateCorners(iftImage *plate)
{
    iftAdjRel *A = iftCircular(1.0);
    iftSet *S = iftObjectBorderSet(plate, A), *Saux1, *Saux2;
    iftPoint *P = (iftPoint *) calloc(4,sizeof(iftPoint));

    iftDestroyAdjRel(&A);

    int dmax = INFINITY_INT_NEG;

    Saux1 = S;
    while (Saux1 != NULL) {
        int p1 = Saux1->elem;
        iftVoxel u1 = iftGetVoxelCoord(plate,p1);
        Saux2 = Saux1;
        while (Saux2 != NULL) {
            int p2 = Saux2->elem;
            iftVoxel u2 = iftGetVoxelCoord(plate,p2);
            int dist = iftSquaredVoxelDistance(u1,u2);
            if (dist > dmax) {
                P[0].x = u1.x;
                P[0].y = u1.y;
                P[1].x = u2.x;
                P[1].y = u2.y;
                dmax   = dist;
            }
            Saux2 = Saux2->next;
        }
        Saux1 = Saux1->next;
    }


    /* Compute the normal vector to P[0]P[1] */

    iftVector N, U;

    U.z = N.z = 0;
    U.x = (P[1].x - P[0].x);
    U.y = (P[1].y - P[0].y);
    U   = iftNormalizeVector(U);
    N.x = -U.y; N.y = U.x;

    /* Find the two other corners at the positive and negative sides of
       the line P[0]P[1] */

    float Dmax=INFINITY_FLT_NEG, Dmin=INFINITY_FLT;

    Saux1 = S;
    while (Saux1 != NULL) {
        int p1 = Saux1->elem;
        iftVoxel  u1 = iftGetVoxelCoord(plate,p1);
        iftVector V1;

        V1.z = 0;
        V1.x = u1.x - P[0].x;
        V1.y = u1.y - P[0].y;

        float dist = iftInnerProduct(V1,N);

        if (dist < Dmin) {
            Dmin = dist;
            P[2].x = u1.x;
            P[2].y = u1.y;
        }
        if (dist > Dmax) {
            Dmax = dist;
            P[3].x = u1.x;
            P[3].y = u1.y;
        }
        Saux1 = Saux1->next;
    }

    /* Sort the points in clockwise order */

    iftPoint Pmean;
    iftPoint *Q = (iftPoint *) calloc(4,sizeof(iftPoint));

    Pmean.x = (P[0].x + P[1].x + P[2].x + P[3].x)/4.0;
    Pmean.y = (P[0].y + P[1].y + P[2].y + P[3].y)/4.0;
    Pmean.z = 0;

    for (int i=0; i < 4; i++) {
        if ((P[i].x < Pmean.x)&&(P[i].y < Pmean.y)){
            Q[0].x = P[i].x;
            Q[0].y = P[i].y;
        }
        if ((P[i].x > Pmean.x)&&(P[i].y < Pmean.y)){
            Q[1].x = P[i].x;
            Q[1].y = P[i].y;
        }
        if ((P[i].x > Pmean.x)&&(P[i].y > Pmean.y)){
            Q[2].x = P[i].x;
            Q[2].y = P[i].y;
        }
        if ((P[i].x < Pmean.x)&&(P[i].y > Pmean.y)){
            Q[3].x = P[i].x;
            Q[3].y = P[i].y;
        }
    }

    // print 4 points
    iftImage *points = iftCreateImage(plate->xsize, plate->ysize, plate->zsize);
    int p;
    iftVoxel u;
    for (int j = 0; j < 4 ; ++j) {
        u.x = P[j].x;
        u.y = P[j].y;
        u.z = 0;
        p = iftGetVoxelIndex(plate, u);
        points->val[p] = 255;
    }

    if (WRITE_TEMP_IMGS)
        iftWriteImageP2(points, "points.pgm");

    free(P);
    iftDestroySet(&S);
    return(Q);
}


iftImage* correctPlate(iftImage *orig, iftImage *label) {

    /* Find the four corners */

    iftPoint *src = iftTheFourPlateCorners(label);
    
    iftPoint *dst = (iftPoint *) calloc(4,sizeof(iftPoint));

    dst[0].x = 0;
    dst[0].y = 0;
    dst[1].x = 99;
    dst[1].y = 0;
    dst[2].x = 99;
    dst[2].y = 39;
    dst[3].x = 0;
    dst[3].y = 39;

    iftImage *cimg = iftCorrectImage(orig,src,dst,100,40);
    //free(src);
    return cimg;
}

iftImage* correctPlateF(iftImage *origImg, iftImage *filtImg) {

    // Extract plate mask and find the rotation by PCA
    iftImage *label, *orig;
    iftBoundingBox mbb = iftMinObjectBoundingBox(filtImg, 1, NULL);
    iftVoxel    pos = mbb.begin;
    iftImage   *plate=iftExtractROI(filtImg, mbb);
    iftMatrix  *T = iftObjectAlignMatrixByPCA(plate);

    label = iftTransformImage(plate,T,IFT_TRANSFORMATION_MATRIX);

    // Extract the same ROI from the original image

    iftVoxel    end;
    end.x = pos.x + plate->xsize - 1;
    end.y = pos.y + plate->ysize - 1;
    end.z = 0;
    iftBoundingBox bb = {.begin = pos, .end = end};
    iftImage   *roi=iftExtractROI(origImg, bb);

    // Apply rotation correction

    orig  = iftTransformImage(roi,T,IFT_TRANSFORMATION_MATRIX);
    iftDestroyMatrix(&T);
    iftDestroyImage(&roi);

    // Apply shear correction and find the final ROI

    iftImage *origc;
    iftVoxel v;
    iftVoxel vo, vf, u;

    origc  = iftCreateImage(orig->xsize,orig->ysize,orig->zsize);

    u.z = v.z = 0;
    vo.x = vo.y = INFINITY_INT;
    vf.x = vf.y = INFINITY_INT_NEG;
    vo.z = vf.z = 0;
    for (u.y=0; u.y < label->ysize; u.y++) {
        v.x = 0;
        for (u.x=0; u.x < label->xsize; u.x++) {
            int p = iftGetVoxelIndex(label,u);
            if (label->val[p]) {
                v.y   = u.y;
                int q = iftGetVoxelIndex(label,v);
                origc->val[q]  = orig->val[p];
                if (vo.x > v.x)
                    vo.x = v.x;
                if (vo.y > v.y)
                    vo.y = v.y;
                if (vf.x < v.x)
                    vf.x = v.x;
                if (vf.y < v.y)
                    vf.y = v.y;
                v.x++;
            }
        }
    }

    // Extract final ROI and scale image to a normalized size

    iftDestroyImage(&plate);
    bb.begin = vo;
    bb.end = vf;
    plate = iftExtractROI(origc,bb);
    iftDestroyImage(&origc);
    float sx = 200.0/plate->xsize;
    float sy = 50.0/plate->ysize;
    origc    = iftScaleImage2D(plate,sx,sy);

    iftDestroyImage(&orig);
    iftDestroyImage(&label);

    return origc;
}


iftMatrix *iftInvertMatrixAw(iftMatrix *A)
{
    iftMatrix *B=NULL,*C=NULL;
    double det;
    int c,r;

    if (A->ncols != A->nrows)
        iftError("Matrix is not square","iftInvertMatrix");

    det = iftMatrixDeterminant(A);

    if (fabs(det) < Epsilon){
        printf("%f\n", fabs(det));
    }else{
        C = iftCoFactorMatrix(A);
        B = iftTransposeMatrix(C); /* Adjoint Matrix */
        for (r=0; r < B->nrows; r++)
            for (c=0; c < B->ncols; c++){
                B->val[iftGetMatrixIndex(B,c,r)] /= det;
            }
    }
    iftDestroyMatrix(&C);

    return(B);
}
