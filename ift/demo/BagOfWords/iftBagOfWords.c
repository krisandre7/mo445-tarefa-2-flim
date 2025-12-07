//
// Created by Alan Peixinho on 6/12/15.
//

#include <ift.h>

#define PATCH_SIZE 0
#define DICT_SIZE 1
#define SAMPLE_NUMBER 2
#define POOLING_CODING 3
#define STRIDE_SIZE 4

iftBowCodingPooling CodingPoolingStrategies[] = {iftBowSoftCoding, iftBowHardCoding, iftBowHVSumPooling};

typedef struct ift_bow_problem {
    int imgSize;
    iftBowKernelsEstimator estimator;
    iftPatchesSampler sampler;
    iftPatchFeatsExtractor featsExtractor;
    iftSampler* classificationSampler;
} iftBowProblem;

iftBagOfFeatures* iftCreateBowFromParams(int imgSize, iftBowKernelsEstimator estimator, iftPatchesSampler sampler, iftPatchFeatsExtractor featsExtractor, double *params) {

  iftBowCodingPooling codingPooling = CodingPoolingStrategies[(int)params[POOLING_CODING]];

  return iftCreateBow(FALSE, imgSize, imgSize, 1, params[PATCH_SIZE], params[PATCH_SIZE], 1, params[STRIDE_SIZE], params[STRIDE_SIZE], 1, sampler,
                        featsExtractor, estimator, codingPooling, params[SAMPLE_NUMBER], params[DICT_SIZE]);
}

double iftBowEval(void *problem, iftFileSet *fileSet, double *params) {

  iftBowProblem* bowProblem = (iftBowProblem*) problem;
  iftBagOfFeatures* bow = iftCreateBowFromParams(bowProblem->imgSize, bowProblem->estimator, bowProblem->sampler, bowProblem->featsExtractor, params);

  printf("[ ");
  for (int i = 0; i < 5; ++i) {
    printf("%lf ", params[i]);
  }
  printf("]\n");
  
  if(iftBowSetup(bow)) {
    return 0.0;
  }

  iftBowLearn(bow, fileSet);
  
  printf("Extracting features ...\n");
  iftDataSet* Z = iftBowExtractFeaturesBatch(bow, fileSet);
  
  printf("Train SVM ...\n");
  iftSVM* svm = iftCreateLinearSVC(1e5);
  
  double acc = 0.0;
  
  int niters = bowProblem->classificationSampler->niters;
  for(int it = 0; it < niters; ++it) {
    iftSampleDataSet(Z, bowProblem->classificationSampler, it);
    iftSVMTrainOVA(svm, Z, NULL);
    iftSVMClassifyOVA(svm, Z, IFT_TEST);
    acc += iftTruePositives(Z)/niters;
  }
  
    iftDestroyDataSet(&Z);
    iftDestroySVM(svm);
    iftDestroyBow(&bow);
    return acc;
}

int main(int argc, char** argv)  {
    if(argc<5) {
        iftError("Usage: <dataset path> <image size> <train percentage> <output file>", "iftBagOfWords.c");
        return 1;
    }

    iftFileSet* files;
    iftFileSet* trainFiles;

    iftParamOptimizer* opt;
    iftBowProblem* problem;

    int imgSize, patchSize;
    float trainperc;
    iftSampler* trainSampler;
    iftSampler* learnSampler;

    sscanf(argv[2], "%d", &imgSize);
    sscanf(argv[3], "%f", &trainperc);

    printf(" Ok.\n");

    files = iftLoadFileSetFromDirBySuffix(argv[1], "png");

    int* labels = iftFileSetLabels(files);

    trainSampler = iftStratifiedRandomSubsampling(labels, files->n, 1, files->n * trainperc);
    iftSampleFileSet(files, trainSampler, 0);
    trainFiles = iftExtractFileSamples(files, IFT_TRAIN);

    problem = (iftBowProblem*) calloc(1, sizeof(iftBowProblem));
    problem->imgSize = imgSize;
    problem->estimator = iftBowKMeansKernels;
    problem->sampler = iftBowPatchesRandomSampler;
    problem->featsExtractor = iftBowPatchRawFeatsExtractor;
    problem->classificationSampler = iftNKFold(trainFiles->n, 5, 2);

    opt = iftCreateParamOptimizer(5);

    opt->paramsSpace[PATCH_SIZE] = iftUniformSearchSpace(5, 11, 2);
    opt->paramsSpace[DICT_SIZE] = iftGeometricSearchSpace(32, 256, 2);
    opt->paramsSpace[SAMPLE_NUMBER] = iftUniformSearchSpace(10, 50, 10);
    opt->paramsSpace[POOLING_CODING] = iftUniformSearchSpace(0, 2, 1);
    opt->paramsSpace[STRIDE_SIZE] = iftUniformSearchSpace(1, 5, 1);

    iftRandomSearchDescriptor(opt, iftBowEval, trainFiles, 100, problem);

    iftBagOfFeatures * bow = iftCreateBowFromParams(problem->imgSize, problem->estimator, problem->sampler, problem->featsExtractor, opt->params);

    printf("Validate params ...");

    iftBowSetup(bow);

    printf("Learning dictionary ...");
    fflush(stdout);

    iftBowLearn(bow, trainFiles);
    printf(" Ok.\n");

    printf("Extracting features ...");
    fflush(stdout);

    iftDataSet* Z2 = iftBowExtractFeaturesBatch(bow, files);

    printf(" Ok.\n");

    iftWriteOPFDataSet(Z2, argv[4]);

    iftDestroySampler(&trainSampler);
    iftDestroyBow(&bow);
    return 0;
}

