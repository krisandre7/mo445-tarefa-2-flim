#include "ift.h"

iftIntArray* trainingSuperpixels(iftCSV* csv) {
    iftIntArray* superpixels = iftCreateIntArray(csv->nrows);
    for (int i = 0; i < superpixels->n; ++i)
        superpixels->val[i] = iftImageSampleId(csv->data[i][0]);
    return superpixels;
}

iftIntArray* labelsSuperpixels(iftCSV* csv) {
    iftIntArray* superpixels = iftCreateIntArray(csv->nrows);
    for (int i = 0; i < superpixels->n; ++i)
        superpixels->val[i] = atoi(csv->data[i][0]);
    return superpixels;
}

iftConvNetwork* createConvNetFromParams(double nlayers,/*number of layers*/
                                        double xsize, double ysize, double zsize, double nbands, /*Input image dimensions*/
                                        double* kernelSize, double *nKernel, /*kernel window size and number of kernels*/
                                        double* activation, double *activationThresh, /*Activation (true/false), threshold activation*/
                                        double* poolingSize, double *strideSize, double* alpha, double* normalizationSize) { /*pooling window size, stride size, normalization window side*/

    iftConvNetwork* convNet = iftCreateConvNetwork(nlayers);

    convNet->input_xsize = (int)xsize;
    convNet->input_ysize = (int)ysize;
    convNet->input_zsize = (int)zsize;
    convNet->input_nbands = (int)nbands;

    for (int l = 0; l < nlayers; ++l) {
        convNet->k_bank_adj_param[l] = (int)kernelSize[l];
        convNet->nkernels[l] = (int)nKernel[l];
        convNet->activ_option[l] = (int)activation[l];
        convNet->activ_thres[l] = (float)activationThresh[l];
        convNet->pooling_adj_param[l] = (int) poolingSize[l];
        convNet->stride[l] = (int)strideSize[l];
        convNet->alpha[l] = (float)alpha[l];
        convNet->norm_adj_param[l] = (int)normalizationSize[l];
    }

    fflush(stdout);

    //The magic happens here
    iftCreateRandomKernelBanks(convNet);
    iftCreateAdjRelAlongNetwork(convNet);

    if(!iftValidArchitecture(convNet)) {
        iftDestroyConvNetwork(&convNet);
        return NULL;
    }

    iftMImageIndexMatrices(convNet);


    return convNet;
}

typedef struct iftConvNetSegmentationProblem{
    iftMImage* mimg;
    iftImage* img;
    iftDataSet* dataset;
    iftImage* superpixels;
    iftIntArray* alllabels;
    iftIntArray* trainingSuperpixels;
} iftConvNetSegmentationProblem;

int nvalid = 0;

void meanResponse(iftDataSet* dataset, int nsuperpixels, iftDataSet* Z) {

    iftIntArray* trueLabel = iftCreateIntArray(nsuperpixels);
    iftIntArray* status = iftCreateIntArray(nsuperpixels);

    iftIntArray* npixels = iftCreateIntArray(nsuperpixels);

    for (int i = 0; i < status->n; ++i) {
        status->val[i] = IFT_UNKNOWN;
    }


    for (int i = 0; i < Z->nsamples; ++i) {
        for (int j = 0; j < Z->nfeats; ++j) {
            Z->sample[i].feat[j] = 0;
        }
    }

    for(int i=0; i<dataset->nsamples; ++i) {
         int sp = dataset->sample[i].id;
         npixels->val[sp]++;
    }

    for (int i = 0; i < dataset->nsamples; ++i) {
        int sp = dataset->sample[i].id;
        for (int j = 0; j < dataset->nfeats; ++j) {
            //Z->sample[sp].feat[featstart+j] = iftMax( dataset->sample[i].feat[j], Z->sample[sp].feat[j] );
            Z->sample[sp].feat[j] += dataset->sample[i].feat[j]/npixels->val[sp];
        }
        trueLabel->val[sp] = dataset->sample[i].truelabel;
        status->val[sp] = dataset->sample[i].status;
    }

    for (int i = 0; i < nsuperpixels; ++i) {
        Z->sample[i].truelabel = trueLabel->val[i];
        Z->sample[i].status = status->val[i];
    }

    iftDestroyIntArray(&npixels);
    iftDestroyIntArray(&status);
    iftDestroyIntArray(&trueLabel);

    printf("Ok\n");
}


void meanResponseStd(iftDataSet* dataset, int nsuperpixels, iftDataSet* Z) {

    iftIntArray* trueLabel = iftCreateIntArray(nsuperpixels);
    iftIntArray* status = iftCreateIntArray(nsuperpixels);

    iftIntArray* npixels = iftCreateIntArray(nsuperpixels);

    for (int i = 0; i < status->n; ++i) {
        status->val[i] = IFT_UNKNOWN;
    }


    for (int i = 0; i < Z->nsamples; ++i) {
        for (int j = 0; j < Z->nfeats; ++j) {
            Z->sample[i].feat[j] = 0;
        }
    }

    for(int i=0; i<dataset->nsamples; ++i) {
        int sp = dataset->sample[i].id;
        npixels->val[sp]++;
    }

    for (int i = 0; i < dataset->nsamples; ++i) {
        int sp = dataset->sample[i].id;
        for (int j = 0; j < dataset->nfeats; ++j) {
            //Z->sample[sp].feat[j] = iftMax( dataset->sample[i].feat[j], Z->sample[sp].feat[j] );
            Z->sample[sp].feat[j] += dataset->sample[i].feat[j]/npixels->val[sp];
        }
        trueLabel->val[sp] = dataset->sample[i].truelabel;
        status->val[sp] = dataset->sample[i].status;
    }

    for (int i = 0; i < dataset->nsamples; ++i) {
        int sp = dataset->sample[i].id;
        for (int j = 0; j < dataset->nfeats; ++j) {
            int featstart = dataset->nfeats;
            float diff = pow(dataset->sample[i].feat[j] - Z->sample[sp].feat[j], 2.0);
            Z->sample[sp].feat[featstart+j] += diff;
        }
    }

    for (int i = 0; i < nsuperpixels; ++i) {
        Z->sample[i].truelabel = trueLabel->val[i];
        Z->sample[i].status = status->val[i];

        for (int j = 0; j < dataset->nfeats; ++j) {
            int featstart = dataset->nfeats;
            Z->sample[i].feat[featstart+j] = sqrt(Z->sample[i].feat[featstart+j]/npixels->val[i]);
        }
    }

    iftDestroyIntArray(&npixels);
    iftDestroyIntArray(&status);
    iftDestroyIntArray(&trueLabel);

    printf("Ok\n");
}


double convNetEvalSup(void* problem, double *theta) {


    double nlayers, xsize, ysize, zsize, nbands;
    double kernelSize[3], nKernels[3], activation[3], activationThresh[3], poolingSize[3], strideSize[3], alpha[3], normalizationSize[3];

    iftConvNetSegmentationProblem* convNetSegmProb = (iftConvNetSegmentationProblem*) problem;

    nlayers = theta[0];

    int idx;
    for (int l = 0; l < nlayers; ++l) {
        idx = 1+l*8;
        kernelSize[l] = theta[idx++];
        nKernels[l] = theta[idx++];
        activation[l] = theta[idx++];
        activationThresh[l] = theta[idx++];
        poolingSize[l] = theta[idx++];
        strideSize[l] = theta[idx++];
        alpha[l] = theta[idx++];
        normalizationSize[l] = theta[idx++];
    }

    xsize = convNetSegmProb->mimg->xsize;
    ysize = convNetSegmProb->mimg->ysize;
    zsize = convNetSegmProb->mimg->zsize;

    nbands = convNetSegmProb->mimg->m;

    iftConvNetwork* convNetwork = createConvNetFromParams(nlayers, xsize, ysize, zsize, nbands,
                                                          kernelSize, nKernels, activation, activationThresh,
                                                          poolingSize, strideSize, alpha, normalizationSize);

    if(convNetwork == NULL) {
        printf("Bad Architecture!\n");
        return 0.0;
    }

    iftPrintConvNetArch(convNetwork);

    iftMImage* mimg = iftApplyConvNetwork(convNetSegmProb->mimg, convNetwork);

//iftMImage* mimg = convNetSegmProb->mimg;

//    iftMImage* mimg = iftResizeMImage(convNetSegmProb->mimg, 348, 344, 1);

    iftMImage* resimg = iftResizeMImage(mimg, convNetwork->input_xsize, convNetwork->input_ysize, convNetwork->input_zsize);

    iftDestroyMImage(&mimg);
    mimg = resimg;

    iftWriteConvNetwork(convNetwork, "isf2_.convnet");
    iftDestroyConvNetwork(&convNetwork);

    int nsuperpixels = iftMaximumValue(convNetSegmProb->superpixels);

    iftBMap* isTraining = iftCreateBMap(nsuperpixels);

    int ntrainSamples = 0;
    for (int p = 0; p < convNetSegmProb->superpixels->n; ++p) {
        int sp = convNetSegmProb->superpixels->val[p];
        if (iftFindIntArrayElementIndex(convNetSegmProb->trainingSuperpixels->val, convNetSegmProb->trainingSuperpixels->n, sp) != IFT_NIL ) {
            ntrainSamples++;
            iftBMapSet1(isTraining, sp-1);
        }
    }

    iftDataSet* fulldataset = iftMImageToDataSet(mimg);

    iftDestroyMImage(&mimg);
    printf("NSamples: %d\n", fulldataset->nsamples);

    for (int p = 0; p < fulldataset->nsamples; ++p) {
        int sp = convNetSegmProb->superpixels->val[p];
        fulldataset->sample[p].id = sp - 1;
        fulldataset->sample[p].truelabel = convNetSegmProb->alllabels->val[sp-1];
        if(iftBMapValue(isTraining, sp-1)) {
            fulldataset->sample[p].status = IFT_TRAIN;
        }
        else {
            fulldataset->sample[p].status = IFT_UNKNOWN;
        }

    }

    iftCountNumberOfClassesDataSet(fulldataset);

    printf("(%d, %d) => ", fulldataset->nsamples, fulldataset->nfeats);

    iftIntArray* classCount = iftCreateIntArray(fulldataset->nclasses);

    iftDataSet* Z = iftCreateDataSet(nsuperpixels, fulldataset->nfeats);

    for (int i = 0; i < convNetSegmProb->trainingSuperpixels->n; ++i) {
        iftBMapSet1(isTraining, convNetSegmProb->trainingSuperpixels->val[i]-1);
    }

    for (int i = 0; i < nsuperpixels; ++i) {
        Z->sample[i].truelabel = convNetSegmProb->alllabels->val[i];
    }

    iftSetStatus(Z, IFT_UNKNOWN);
    iftCountNumberOfClassesDataSet(Z);

    meanResponse(fulldataset, nsuperpixels, Z);

    for (int c = 1; c <= fulldataset->nclasses ; ++c) {
        classCount->val[c-1] = iftCountSamplesTrueLabel(fulldataset, c, IFT_TRAIN);
        printf("C%d = %d - %d\n", c, classCount->val[c-1], iftCountSamplesLabel(fulldataset, c, IFT_TRAIN));
    }

    printf("Get labels ... \n");
    iftIntArray* labels = iftGetDataSetTrueLabels(Z);

    printf("Labels - %d\n", (int)labels->n);

    iftSampler* sampler = iftStratifiedRandomSubsampling(labels->val, labels->n, 10, labels->n*0.5);

    iftSVM* svm = iftCreatePreCompSVC(1e5);

    iftFloatArray* accs = iftCreateFloatArray(sampler->niters);

    printf("Evaluating ...\n");
    iftDestroyDataSet(&fulldataset);

    for (int it = 0; it < sampler->niters; ++it) {
        iftSampleDataSet(Z, sampler, it);

        for (int c = 1; c <= Z->nclasses ; ++c) {
            classCount->val[c-1] = iftCountSamplesTrueLabel(Z, c, IFT_TEST);
            printf("Before classify %d = %d - %d\n", c, classCount->val[c-1], iftCountSamplesLabel(Z, c, IFT_TEST));
        }

        printf("\n=====================\n");


        printf("Extract train samples\n");
        float trace;
        iftDataSet* Zk = iftKernelizeDataSet(Z, Z, LINEAR, true, &trace);
        printf("Train\n");
        iftSVMTrainOVA(svm, Zk, NULL);
        iftSVMClassifyOVA(svm, Zk, IFT_TEST);

        for (int c = 1; c <= Z->nclasses ; ++c) {
            classCount->val[c-1] = iftCountSamplesTrueLabel(Zk, c, IFT_TEST);
            printf("After classify %d = %d - %d\n", c, classCount->val[c-1], iftCountSamplesLabel(Zk, c, IFT_TEST));
        }

        accs->val[it] = iftTruePositives(Zk);
        printf("Acc[%d] = %f\n", it, accs->val[it]);


        printf("Compute acc ...\n");
        accs->val[it] = iftNormAccuracy(Zk);
        printf("Nacc[%d] = %f\n", it, accs->val[it]);

	    iftDestroyDataSet(&Zk);

    }

    float meanAcc = iftMean(accs->val, accs->n);

    printf("Acc = %f\n", meanAcc);

    nvalid++;

    iftDestroySVM(svm);
    iftDestroyDataSet(&fulldataset);
    iftDestroySampler(&sampler);
    iftDestroyFloatArray(&accs);
    iftDestroySVM(svm);

    return meanAcc;
}

void iftWriteDatasetCSV(iftDataSet* Z, const char* filename);

double convNetEvalTest(void* problem, iftConvNetwork* convNetwork) {

    iftConvNetSegmentationProblem* convNetSegmProb = (iftConvNetSegmentationProblem*) problem;

    if(convNetwork == NULL) {
        printf("Bad Architecture!\n");
        return 0.0;
    }

    iftPrintConvNetArch(convNetwork);

    iftMImage* mimg = iftApplyConvNetwork(convNetSegmProb->mimg, convNetwork);

    iftMImage* resimg = iftResizeMImage(mimg, convNetwork->input_xsize, convNetwork->input_ysize, convNetwork->input_zsize);

    iftDestroyMImage(&mimg);
    mimg = resimg;

    int nsuperpixels = iftMaximumValue(convNetSegmProb->superpixels);

    iftDataSet* dataset = iftCreateDataSet(mimg->n, mimg->m);
    iftDataSet* Z = iftCreateDataSet(nsuperpixels, mimg->m);

    iftBMap* isTraining = iftCreateBMap(nsuperpixels);

    for (int i = 0; i < convNetSegmProb->trainingSuperpixels->n; ++i) {
        iftBMapSet1(isTraining, convNetSegmProb->trainingSuperpixels->val[i]-1);
    }

    for (int i = 0; i < nsuperpixels; ++i) {
        Z->sample[i].truelabel = convNetSegmProb->alllabels->val[i];
    }

    int ntrainsamples = 0;

    for (int p = 0; p < convNetSegmProb->superpixels->n; ++p) {
        int sp = convNetSegmProb->superpixels->val[p];
         for (int b = 0; b < mimg->m; ++b) {
            dataset->sample[p].feat[b] = mimg->val[p][b];
        }
        dataset->sample[p].truelabel = convNetSegmProb->alllabels->val[sp-1];
        dataset->sample[p].id = sp-1;
        if(iftBMapValue(isTraining, sp-1)) {
            dataset->sample[p].status = IFT_TRAIN;
            ntrainsamples++;
            Z->sample[sp-1].status=IFT_TRAIN;
        }
    }

    dataset->ntrainsamples = ntrainsamples;
    iftCountNumberOfClassesDataSet(dataset);
    Z->ntrainsamples = iftCountSamples(Z, IFT_TRAIN);
    iftCountNumberOfClassesDataSet(Z);
    iftIntArray* classCount = iftCreateIntArray(dataset->nclasses);

    for (int c = 1; c <= dataset->nclasses ; ++c) {
        classCount->val[c-1] = iftCountSamplesTrueLabel(dataset, c, IFT_TRAIN);
        printf("C%d = %d\n", c, classCount->val[c-1]);
    }

    for (int c = 1; c <= dataset->nclasses ; ++c) {
        classCount->val[c-1] = iftCountSamplesTrueLabel(dataset, c, IFT_TRAIN);
        printf("C%d = %d\n", c, classCount->val[c-1]);
    }

    printf("(%d, %d) => ", dataset->nsamples, dataset->nfeats);
    iftDestroyMImage(&mimg);

    meanResponse(dataset, nsuperpixels, Z);

    iftSVM* svm = iftCreatePreCompSVC(1e5);

    float trace;
    iftDataSet* Zk = iftKernelizeDataSet(Z, Z, LINEAR, true, &trace);

    iftSVMTrainOVO(svm, Zk);
    iftSVMClassifyOVO(svm, Zk, IFT_TEST);

    for(int s = 0; s<Z->nsamples; ++s) {
        Z->sample[s].label = Zk->sample[s].label;
    }

    for (int c = 1; c <= dataset->nclasses ; ++c) {
        classCount->val[c-1] = iftCountSamplesTrueLabel(Z, c, IFT_TEST);
        printf("After classify %d = %d - %d\n", c, classCount->val[c-1], iftCountSamplesLabel(Z, c, IFT_TEST));
    }

    float acc = iftTruePositives(Z);

    printf("Acc = %f\n", acc);
    printf("NACC = %f\n", iftNormAccuracy(Z));


    for (int c = 1; c <= dataset->nclasses ; ++c) {
        int count = iftCountSamplesLabel(Z, c, IFT_TEST);
        int counttrue = iftCountSamplesTrueLabel(Z, c, IFT_TEST);
        printf("After voting %d = %d - %d\n", c, counttrue, count);
    }

    printf("========================\n");

    for (int c = 1; c <= dataset->nclasses ; ++c) {
        int count = iftCountSamplesLabel(Z, c, IFT_TRAIN);
        int counttrue = iftCountSamplesTrueLabel(Z, c, IFT_TEST);
        printf("C%d = %d - %d\n", c, count, counttrue);
    }

    printf("(%d, %d)", dataset->nsamples, dataset->ntrainsamples);
    printf("(%d, %d)", Z->nsamples, Z->ntrainsamples);

    printf("Mimg = (%d, %d)", convNetSegmProb->mimg->xsize, convNetSegmProb->mimg->ysize);

    acc = iftTruePositives(Z);
    printf("Acc = %f\n", acc);

    printf("NACC = %f\n", iftNormAccuracy(Z));

    iftWriteDatasetCSV(Z, "convnet.csv");


    iftDestroySVM(svm);
    iftDestroyDataSet(&dataset);
    iftDestroyMImage(&mimg);
    iftDestroyConvNetwork(&convNetwork);
    iftDestroyDataSet(&Z);

    return acc;
}

void iftWriteDatasetCSV(iftDataSet* Z, const char* filename) {

    FILE* fp = fopen(filename, "wb");

    for (int i = 0; i < Z->nsamples; ++i) {
        for (int j = 0; j < Z->nfeats; ++j) {
            if(j>0)
                fprintf(fp, ", ");
            fprintf(fp, "%f", Z->sample[i].feat[j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

double datasetEval(iftConvNetSegmentationProblem *problem) {

    float acc;

    int nsuperpixels = iftMaximumValue(problem->superpixels);

    iftDataSet* Z = iftCreateDataSet(nsuperpixels, 3);
    iftIntArray* npixels = iftCreateIntArray(nsuperpixels);

    iftDataSet* Z2 = problem->dataset;

    iftDataSet* Zp = iftMImageToDataSet(problem->mimg);

    for (int p = 0; p < problem->mimg->n; ++p) {
        int sp = problem->superpixels->val[p];
        Z->sample[sp-1].id = sp;

        iftColor rgb;
        rgb.val[0] = problem->mimg->val[p][0];
        rgb.val[1] = problem->mimg->val[p][1];
        rgb.val[2] = problem->mimg->val[p][2];

        iftFColor lab = iftRGBtoLab(rgb, 255);

        lab.val[0] = (lab.val[0] - 0.)/(99.998337 - 0.);
        lab.val[1] = (lab.val[1] + 86.182236)/(98.258614 + 86.182236);
        lab.val[2] = (lab.val[2] + 107.867744)/(94.481682 + 107.867744);

        for (int b = 0; b < 3; ++b) {
            Z->sample[sp-1].feat[b] += lab.val[b];
        }

        Z2->sample[sp-1].truelabel = problem->alllabels->val[sp-1];
        Z2->sample[sp-1].status = IFT_TEST;

        Z->sample[sp-1].truelabel = problem->alllabels->val[sp-1];
        Z->sample[sp-1].status = IFT_TEST;

        Zp->sample[p].truelabel = problem->alllabels->val[sp-1];
        Zp->sample[p].status = IFT_TEST;
        Zp->sample[p].id = sp-1;

        npixels->val[sp-1]++;
    }

    for (int p = 0; p < nsuperpixels; ++p) {
        for (int b = 0; b < 3; ++b) {
            Z->sample[p].feat[b]/=npixels->val[p];
        }
    }

    iftBMap* isTraining = iftCreateBMap(nsuperpixels);

    for (int i = 0; i < problem->trainingSuperpixels->n; ++i) {
        Z2->sample[problem->trainingSuperpixels->val[i]-1].status = IFT_TRAIN;
        Z->sample[problem->trainingSuperpixels->val[i]-1].status = IFT_TRAIN;
        iftBMapSet1(isTraining, problem->trainingSuperpixels->val[i]-1);
    }

    for (int s = 0; s < Zp->nsamples; ++s) {
        if(iftBMapValue(isTraining, Zp->sample[s].id)) {
            Zp->sample[s].status = IFT_TRAIN;
        }
    }

    Z2->ntrainsamples = iftCountSamples(Z2, IFT_TRAIN);
    Z2->nclasses = iftCountNumberOfClassesDataSet(Z2);

    Z->ntrainsamples = iftCountSamples(Z, IFT_TRAIN);
    Z->nclasses = iftCountNumberOfClassesDataSet(Z);


    Zp->ntrainsamples = iftCountSamples(Zp, IFT_TRAIN);
    Zp->nclasses = iftCountNumberOfClassesDataSet(Zp);

    iftSVM* svm = iftCreateRBFSVC(1e5, 1.0);
    iftSVMTrainOVA(svm, Z2, NULL);
    iftSVMClassifyOVA(svm, Z2, IFT_TEST);

    acc = iftTruePositives(Z2);
    printf("ACC = %f\n", acc);
    acc = iftNormAccuracy(Z2);
    printf("NACC = %f\n", acc);


    return acc;
}

iftDataSet* meanPixelDataset(iftCSV* csv) {
    int nsamples = csv->nrows;
    int nfeats = csv->ncols;

    iftDataSet* Z = iftCreateDataSet(nsamples, nfeats);

    for (int s = 0; s < Z->nsamples; ++s) {
        for (int f = 0; f < Z->nfeats; ++f) {
            Z->sample[s].feat[f] = atof(csv->data[s][f]);
        }
    }

    return Z;
}

int main(int argc, const char** argv) {

    iftCSV* training = iftReadCSV("/home/peixinho/Projects/Rome/Bases/train3.txt", ' ');
    iftCSV* testlabels = iftReadCSV("/home/peixinho/Projects/Rome/Bases/Rome/labels.csv", ' ');
    iftCSV* meanPixel = iftReadCSV("/home/peixinho/descriptors_rome/descritores_single_superpixel/rome_mc.csv", ',');

    iftImage* img = iftReadImageByExt("/home/peixinho/Projects/Rome/Bases/Rome/rome.ppm");

//    iftConvNetwork* convnet = iftReadConvNetwork("isf_best_mean_svm.convnet");

    iftConvNetSegmentationProblem* problem = (iftConvNetSegmentationProblem*) calloc(1, sizeof(iftConvNetSegmentationProblem));

    problem->img = img;
    problem->mimg = iftImageToMImage(img, RGB_CSPACE);
    for(int b = 0; b<problem->mimg->m; ++b) {
        for(int p = 0; p<problem->mimg->n; ++p) {
            problem->mimg->val[p][b] -= 127.0;
        }
    }
    problem->superpixels = iftReadImageByExt("/home/peixinho/Projects/Rome/Bases/Rome/rome.pgm");
    problem->dataset = meanPixelDataset(meanPixel);
    problem->trainingSuperpixels = trainingSuperpixels(training);
    problem->alllabels = labelsSuperpixels(testlabels);

    iftParamOptimizer* opt = iftReadParamOptimizer("../demo/Superpixels/iftConvNetOpt.json");
    iftRandomSearch(opt, convNetEvalSup, 2000, problem);

    convNetEvalSup(problem, opt->params);

//    convNetEval2(problem, convnet);

    iftDestroyIntArray(&(problem->alllabels));
    iftDestroyIntArray(&(problem->trainingSuperpixels));
    iftDestroyMImage(&(problem->mimg));
    iftDestroyImage(&img);
    free(problem);

    return 0;
}
