#include "ift.h"
#include "iftExtractFeatures.h"
#define BOX_SIZE_X 100
#define BOX_SIZE_Y 45

#define DESC_FREQ_X 35
#define DESC_FREQ_Y 15

#define FEATS_SIZE (bow->nfeats)

#define MIN_CH 0
#define NORM_VAL_FEAT_CH 1
/*
#define MIN_CH 400
#define NORM_VAL_FEAT_CH 100
*/

/* Feature extraction functions */
iftImage *enhanceImage(iftImage *orig);
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
void addValueToBoundingBox(iftImage *label, iftVoxel u, iftVoxel v);

/* Train sample selection functions */
void selectTrainSamples(iftImage *orig, iftImage *label, iftImage *gradientImg, int nsamples, iftImage *compX, iftImage *compY, iftDataSet *Z, int indexZ);
//void selectTrainSamplesRD(iftImage *orig, iftImage *label, iftImage *gradientImg, int nsamples, iftDataSet *Z, int indexZ);
int isCandidateSample(iftImage *compX, iftImage *compY, iftVoxel u, iftVoxel v);
int randInterval(int min, int max);
int isAvailableVoxel(iftImage *label, iftVoxel u, iftVoxel v);

/* Detect plates in new images */
iftImage *detectPlate(iftImage *orig, iftImage *label, iftSVM* svm, char *filename);

iftBow* bow = NULL;


int main(int argc, char *argv[]) {

    iftImage* trainOrig, *trainLabel, *testOrig, *testLabel, *gradientImg, *compX, *compY, *testResult;
    iftDataSet *Z, *Zn;
    int indexZ, samplesPerImage;
    timer *t1 = NULL, *t2 = NULL;
    char outfile[100], outFileTest[100],outfileGradient[100];

    /*--------------------------------------------------------*/
    /*
    int MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();
    */
    /*--------------------------------------------------------*/

    if (argc != 6) {
        fprintf(stdout, "Usage: iftSelectCandidatesFreq <train_orig_folder> <train_label_folder> <test_orig_folder> <output_folder>\n");
        fprintf(stdout, "       train input images folder:      path to original grayscale images\n");
        fprintf(stdout, "       train label images folder:      path to label images\n");
        fprintf(stdout, "       test input images folder:       path to original grayscale images\n");
        fprintf(stdout, "       test label images folder:       path to label grayscale images\n");
        fprintf(stdout, "       output candidate folder: path to binary images with candidates for plate location\n");
        exit(1);
    }

    iftFileSet* bowTrain = iftLoadFileSetFromDirByExt("/home/peixinho/TrainPlates/trainpatches", "pgm");

    iftFileSet* trainOrigDir = iftLoadFileSetFromDirByExt(argv[1], "pgm");
    iftFileSet* trainLabelDir = iftLoadFileSetFromDirByExt(argv[2], "pgm");
    iftFileSet* testOrigDir = iftLoadFileSetFromDirByExt(argv[3], "pgm");
    iftFileSet* testLabelDir = iftLoadFileSetFromDirByExt(argv[4], "pgm");
    samplesPerImage = 15;


    bow = iftCreateBow(FALSE, 100, 45, 1,
    30, 15, 1,
    5, 2, 1, iftBowPatchesRandomSampler,
            iftBowPatchRawFeatsExtractor, iftBowKMeansKernels,
            iftBowSoftCoding, 150, 32);

    iftBowSetup(bow);
    iftBowLearn(bow, bowTrain);

    //Z = iftCreateDataSet(samplesPerImage*(trainOrigDir->nfiles), (DESC_FREQ_X + DESC_FREQ_Y) );
    Z = iftCreateDataSet(samplesPerImage*(trainOrigDir->nfiles), FEATS_SIZE);
    Z->nclasses = 2;
    Z->ntrainsamples = 0;
    // Train samples
    indexZ = 0;
    for (int i = 0; i < trainOrigDir->nfiles; ++i) {

        trainOrig = iftReadImageP5(trainOrigDir->files[i]->pathname);
        trainLabel = iftReadImageP5(trainLabelDir->files[i]->pathname);

        t1 = iftTic();
        gradientImg = enhanceImage(trainOrig);

        t2 = iftToc();

        compX = countComponentsX(gradientImg);
        compY = countComponentsY(gradientImg);
        selectTrainSamples(trainOrig, trainLabel, gradientImg, samplesPerImage, compX, compY, Z, indexZ);

        indexZ += samplesPerImage;

        fprintf(stdout, "%dth extract features in %f ms\n", i+1,  iftCompTime(t1, t2));
        sprintf(outfile, "%s%s", argv[5], iftFilename(trainOrigDir->files[i]->pathname, NULL));
        printf("%s\n", outfile);
        iftWriteImageP2(gradientImg, outfile);

        iftDestroyImage(&trainOrig);
        iftDestroyImage(&gradientImg);
        iftDestroyImage(&compX);
        iftDestroyImage(&compY);
    }
    /* Normalize dataset */
    //Zn = iftNormalizeDataSetByGlobalMax(Z);
    Zn = iftCopyDataSet(Z);
    iftDestroyDataSet(&Z);
    iftSetStatus(Zn, IFT_TRAIN);

    // Train Classifier
    printf("SVM train\n");
    //iftSVM* svm = iftCreateLinearSVC(1e4);
    iftSVM* svm = iftCreateRBFSVC(1e5, 0.1);
    iftSVMTrainOVO(svm, Zn);

    // Apply classifier
    for (int i = 0; i < testOrigDir->nfiles; ++i) {

        testOrig = iftReadImageP5(testOrigDir->files[i]->pathname);
        testLabel = iftReadImageP5(testLabelDir->files[i]->pathname);

        printf("Test image %s\n", testOrigDir->files[i]->pathname);

        gradientImg = enhanceImage(testOrig);

        sprintf(outFileTest, "%s%s", argv[5], iftFilename(testOrigDir->files[i]->pathname, NULL));
        testResult = detectPlate(testOrig, testLabel, svm, outFileTest);

        sprintf(outfileGradient, "%s%s.gra.pgm", argv[5], iftFilename(testOrigDir->files[i]->pathname, NULL));
        printf("%s\n", outfileGradient);
        iftWriteImageP2(gradientImg, outfileGradient);

        iftDestroyImage(&gradientImg);
        iftDestroyImage(&testOrig);
        iftDestroyImage(&testLabel);
        iftDestroyImage(&testResult);
    }

    iftDestroyFileSet(&trainOrigDir);
    iftDestroyFileSet(&trainLabelDir);
    iftDestroyFileSet(&testOrigDir);

    /* ---------------------------------------------------------- */
    /*
    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial != MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
               MemDinInicial, MemDinFinal);
    */

    return (0);
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
        //printf("%f\n", lbp->val[i]);
    }
    //iftWriteImageP2(lbpImg, "bb_lbp.pgm");
    iftDestroyAdjRel(&A);
    return lbp;
}

//void detectPlate(iftImage *orig, iftImage *label, iftCplGraph* graph) {
iftImage *detectPlate(iftImage *orig, iftImage *label, iftSVM* svm, char *filename) {
    iftDataSet *Z;
    iftFeatures *feat;
    iftImage *gradientImg, *compX, *compY, *candImg, *normCandImg, *boxImg;
    iftVoxel u,v;
    char binOutFile[100], fuzzyOutFile[100];
    int i, j, strideX, strideY, numCand, nsamples, indexZ;
    strideX = 9;
    strideY = 6;
    u.z = 0;
    v.z = 0;
    numCand = 0;

    /* Extract features */
    gradientImg = enhanceImage(orig);
    compX = countComponentsX(gradientImg);
    compY = countComponentsY(gradientImg);

    nsamples = ((orig->xsize - BOX_SIZE_X)/strideX) * ((orig->ysize - BOX_SIZE_Y)/strideY);

    /* Create dataset  */
    //Z = iftCreateDataSet(nsamples, (DESC_FREQ_X + DESC_FREQ_Y));
    Z = iftCreateDataSet(nsamples, FEATS_SIZE);
    iftSetStatus(Z, IFT_TRAIN);
    candImg = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
    indexZ = 0;
    for (j = 0; j < (orig->xsize - BOX_SIZE_X - strideX + 1); j+=strideX) {
        for (i = 0; i < (orig->ysize - BOX_SIZE_Y - strideY + 1); i += strideY) {
            u.x = j;
            u.y = i;
            v.x = u.x + BOX_SIZE_X - 1;
            v.y = u.y + BOX_SIZE_Y - 1;

            printf("%d, %d\n", i, j);

            //feat = getFreqFeatures(compX, compY, u, v);
            boxImg = createBoundingBox2DByCoord(orig, u, v);

            if(boxImg->xsize!=100 || boxImg->ysize!=45) {
                iftImage* newImage = iftInterp2D(boxImg, 100.0/boxImg->xsize, 45.0/boxImg->ysize);
                iftDestroyImage(&boxImg);
                boxImg = newImage;
            }

            //feat = computeLBP(boxImg);
            //feat = iftExtractBIC(boxImg, 64);
            feat = iftBowExtractFeatures(bow, boxImg);
            iftDestroyImage(&boxImg);

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
    iftSVMClassifyOVO(svm, Z, IFT_TEST);

    /* Write normalized candidate image */
    indexZ = 0;
    for (j = 0; j < orig->xsize - BOX_SIZE_X; j+=strideX) {
        for (i = 0; i < orig->ysize - BOX_SIZE_Y; i += strideY) {
            u.x = j;
            u.y = i;
            v.x = u.x + BOX_SIZE_X - 1;
            v.y = u.y + BOX_SIZE_Y - 1;

            if ( Z->sample[indexZ].label == 1) {
                addValueToBoundingBox(candImg, u, v);
                numCand++;
                /*
                // Print positive samples
                char outfile[100];
                iftImage *bb = createBoundingBox2DByCoord(gradientImg, u, v);
                int randN = randInterval(0, 10000);
                sprintf(outfile, "old/sample_pos_%d.pgm", randN);
                iftWriteImageP5(bb, outfile);
                iftDestroyImage(&bb);
                 */
            }
            indexZ++;
        }
    }
    int maxVal = iftMaximumValue(candImg);
    if (maxVal > 0)
        normCandImg = iftNormalize(candImg, 0, 255);
    else
        normCandImg = iftCopyImage(candImg);

    printf("numCand: %d\n", numCand);
    sprintf(fuzzyOutFile, "%s.fuz.pgm", filename);
    iftWriteImageP2(normCandImg, fuzzyOutFile);

    // Apply otsu threshold
    int t = iftOtsu(normCandImg);
    iftImage *thrImg = iftThreshold(normCandImg, t, 255, 255);
    sprintf(binOutFile, "%s.bin.pgm", filename);
    iftWriteImageP2(thrImg, binOutFile);

    iftDestroyImage(&normCandImg);
    iftDestroyImage(&candImg);
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
    //printf("Total CH: %d\n", (totalX+totalY));
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

    iftVoxel uPlaca, vPlaca, u, v, u_arr[10], v_arr[10];
    int nOtherPositives, nOtherNegatives;
    iftFeatures *feat;
    iftImage *negSampleImg, *boxImg;
    negSampleImg = iftCreateImage(orig->xsize, orig->ysize, orig->zsize);
    // Select positive samples
    getBoundingBoxCoord(label, 1, &uPlaca, &vPlaca);
    //feat = getFreqFeatures(compX, compY, uPlaca, vPlaca);

    boxImg = createBoundingBox2DByCoord(orig, uPlaca, vPlaca);
    //feat = iftExtractBIC(boxImg, 64);
    //feat = computeLBP(boxImg);

    if(boxImg->xsize!=100 || boxImg->ysize!=45) {
        iftImage* newImage = iftInterp2D(boxImg, 100.0/boxImg->xsize, 45.0/boxImg->ysize);
        iftDestroyImage(&boxImg);

        boxImg = newImage;
    }

    feat = iftBowExtractFeatures(bow, boxImg);

    iftWriteImageP5(boxImg, "/home/peixinho/TrainPlates/trainpatches/%d_%d.pgm", 1, indexZ*10+1);

   // assert(feat->n == 512);

    iftDestroyImage(&boxImg);

    // Populate dataset with nunnormalized features
    for (int i = 0; i < feat->n; ++i) {
        Z->sample[indexZ].feat[i] = feat->val[i];
        Z->sample[indexZ].truelabel = 1;
    }
    indexZ++;
    setBoundingBox(label, uPlaca, vPlaca);
    nsamples--;
    iftDestroyFeatures(&feat);

    // Print train positive sample
    //printf("***Select POSITIVE train samples: %f\n", sumFeatureValues(feat));
    /*
    char outfilepos[100];
    int randN = randInterval(0, 10000);
    sprintf(outfilepos, "pos_sample_%d.pgm", randN);
    iftImage *bbpos = iftCreateBoundingBox2D(gradientImg, label, 1);
    iftWriteImageP5(bbpos, outfilepos);
    iftDestroyImage(&bbpos);
    */
    /* Select other positive samples */
    /*
    nOtherPositives = 4;
    u_arr[0].x = uPlaca.x - 3;
    u_arr[0].y = uPlaca.y;
    u_arr[0].z = 0;
    u_arr[1].x = uPlaca.x + 3;
    u_arr[1].y = uPlaca.y;
    u_arr[1].z = 0;

    u_arr[2].x = uPlaca.x;
    u_arr[2].y = uPlaca.y - 3;
    u_arr[2].z = 0;
    u_arr[3].x = uPlaca.x;
    u_arr[3].y = uPlaca.y + 3;
    u_arr[3].z = 0;

    for (int j = 0; j < nOtherPositives; ++j) {
        v_arr[j].x = u_arr[j].x + BOX_SIZE_X - 1;
        v_arr[j].y = u_arr[j].y + BOX_SIZE_Y - 1;
        v_arr[j].z = 0;
        if (iftValidVoxel(label, u_arr[j]) && iftValidVoxel(label, v_arr[j])) {
            feat = getFreqFeatures(compX, compY, u_arr[j], v_arr[j]);
            // Populate dataset with unnormalized features
            for (int i = 0; i < feat->n; ++i) {
                Z->sample[indexZ].feat[i] = feat->val[i];
                Z->sample[indexZ].truelabel = 1;
            }
            indexZ++;
            setBoundingBox(label, u_arr[j], v_arr[j]);
            nsamples--;
            iftDestroyFeatures(&feat);
        }
    }
    */

    //printf("***Select NEGATIVE train samples\n");

    /* select top negative samples */

    nOtherNegatives = 6;
    u_arr[0].x = uPlaca.x - BOX_SIZE_X + 30;
    u_arr[0].y = uPlaca.y - BOX_SIZE_Y;
    u_arr[0].z = 0;
    u_arr[1].x = uPlaca.x - BOX_SIZE_X + 30;
    u_arr[1].y = uPlaca.y - 2 * BOX_SIZE_Y + 20;
    u_arr[1].z = 0;

    u_arr[2].x = uPlaca.x - 30;
    u_arr[2].y = uPlaca.y - BOX_SIZE_Y;
    u_arr[2].z = 0;
    u_arr[3].x = uPlaca.x - 30;
    u_arr[3].y = uPlaca.y - 2 * BOX_SIZE_Y + 20;
    u_arr[3].z = 0;

    u_arr[4].x = uPlaca.x + BOX_SIZE_X - 50;
    u_arr[4].y = uPlaca.y - BOX_SIZE_Y;
    u_arr[4].z = 0;
    u_arr[5].x = uPlaca.x + BOX_SIZE_X - 50;
    u_arr[5].y = uPlaca.y - 2 * BOX_SIZE_Y + 20;
    u_arr[5].z = 0;

    for (int j = 0; j < nOtherNegatives; ++j) {
        v_arr[j].x = u_arr[j].x + BOX_SIZE_X - 1;
        v_arr[j].y = u_arr[j].y + BOX_SIZE_Y - 1;
        v_arr[j].z = 0;
        if (iftValidVoxel(label, u_arr[j]) && iftValidVoxel(label, v_arr[j])) {
            //feat = getFreqFeatures(compX, compY, u_arr[j], v_arr[j]);

            boxImg = createBoundingBox2DByCoord(orig, u_arr[j], v_arr[j]);

            if(boxImg->xsize!=100 || boxImg->ysize!=45) {
                iftImage* newImage = iftInterp2D(boxImg, 100.0/boxImg->xsize, 45.0/boxImg->ysize);
                iftDestroyImage(&boxImg);
                boxImg = newImage;
            }

            feat = iftBowExtractFeatures(bow, boxImg);
            //feat = computeLBP(boxImg);
            //feat = iftExtractBIC(boxImg, 64);

            //iftWriteImageP5(boxImg, "/home/peixinho/TrainPlates/trainpatches/%d_%d.pgm", 2, indexZ*10+2);

            iftDestroyImage(&boxImg);

            // Populate dataset with unnormalized features
            for (int i = 0; i < feat->n; ++i) {
                Z->sample[indexZ].feat[i] = feat->val[i];
                Z->sample[indexZ].truelabel = 2;
            }
            indexZ++;
            setBoundingBox(label, u_arr[j], v_arr[j]);
            nsamples--;
            iftDestroyFeatures(&feat);
        }
    }
    iftWriteImageP2(label, "finalLabel.pgm");


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
                //feat = getFreqFeatures(compX, compY, u, v);
                boxImg = createBoundingBox2DByCoord(orig, u, v);

                if(boxImg->xsize!=100 || boxImg->ysize!=45) {
                    iftImage* newImage = iftInterp2D(boxImg, 100.0/boxImg->xsize, 45.0/boxImg->ysize);
                    iftDestroyImage(&boxImg);
                    boxImg = newImage;
                }

                feat = iftBowExtractFeatures(bow, boxImg);
                //feat = computeLBP(boxImg);
                //feat = iftExtractBIC(boxImg, 64);
                //iftWriteImageP5(boxImg, "/home/peixinho/TrainPlates/trainpatches/%d_%d.pgm", 2, indexZ*10+3);
                iftDestroyImage(&boxImg);

                // Populate dataset with nunnormalized features
                for (int i = 0; i < feat->n; ++i) {
                    Z->sample[indexZ].feat[i] = feat->val[i];
                    Z->sample[indexZ].truelabel = 2;
                }
                indexZ++;
                setBoundingBox(negSampleImg, u, v);
                nsamples--;
                iftDestroyFeatures(&feat);

                // Write bounding box
                /*
                // Print train negative sample
                char outfile[100];
                iftImage *bb = createBoundingBox2DByCoord(gradientImg, u, v);
                int randN = randInterval(0, 10000);
                sprintf(outfile, "neg_sample_%d.pgm", randN);
                iftWriteImageP5(bb, outfile);
                iftDestroyImage(&bb);
                */

            }
        }
    }

    // print selected negative samples in the original image
    char outfileneg[100];
    sprintf(outfileneg, "finalNegLabel%d.pgm", (indexZ/10));
    iftWriteImageP2(negSampleImg, outfileneg);

    iftDestroyImage(&negSampleImg);
}

/*
void selectTrainSamplesRD(iftImage *orig, iftImage *label, iftImage *gradientImg, int nsamples, iftDataSet *Z, int indexZ) {
    iftVoxel uPlaca, vPlaca, u, v, u_arr[10], v_arr[10];
    int nOtherPositives, nOtherNegatives;
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

    //printf("***Select NEGATIVE train samples\n");
    iftWriteImageP2(label, "finalLabel.pgm");


    // Select general negative samples
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
    // print selected negative samples in the original image

    //char outfileneg[100];
    //sprintf(outfileneg, "finalNegLabel%d.pgm", (indexZ/10));
    //iftWriteImageP2(negSampleImg, outfileneg);

    iftDestroyImage(&negSampleImg);

}
*/

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
    //printf("Total CH: %d\n", (totalX + totalY));
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

void addValueToBoundingBox(iftImage *label, iftVoxel u, iftVoxel v) {
    /* increment by 1 pixel values in the label image for the bounding box delimited by u and v (voxels) */
    int p;
    for (int i = u.y; i <= v.y; i++) {
        for (int j = u.x; j <= v.x; j++) {
            p = j + i * label->xsize;
            label->val[p] += 1;
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
    iftDataSet *Zn = iftCopyDataSet(Z);

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
    printf("max: %f\n", max);
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
