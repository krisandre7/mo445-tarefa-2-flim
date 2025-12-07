#include "ift.h"
// TODO Add default svn C and gamma values

int main(int argc, char *argv[])
{
  if (argc < 7) {
    printf("Usage: %s <dataset path> <nBins> <svm C> <svm gamma> <output matrix path> <output trained svm path>", argv[0]);
    return -1;
  }

  iftDataSet *Z = iftReadDataSet(argv[1]);
  int nBins = atoi(argv[2]);
  float C = atof(argv[3]);
  float gamma = atof(argv[4]);
  char *outMxPath = argv[5];
  char *outSvmPath = argv[6];

  // Target results
  iftMatrix *tgt = iftCreateMatrix(Z->nclasses, nBins * 2);
  iftMatrix *binSampleCount = iftCreateMatrix(Z->nclasses, nBins); 
  iftMatrix *binSampleProb = iftCreateMatrixPointer(tgt->val, Z->nclasses, nBins);
  iftMatrix *binErrorCount = iftCreateMatrix(Z->nclasses, nBins);
  iftMatrix *binErrorProb = iftCreateMatrixPointer(&(tgt->val[binSampleFraction->n]), Z->nclasses, nBins);

  // Train/Classify within dataset to obtain error probabilities
  int train_samples = 0.3f * ((float) Z->nsamples);
  iftSampler* sampler = iftRandomSubsampling(Z->nsamples, 1, train_samples);
  iftDataSetSampling(Z, sampler, 0);
  iftDataSet *Z1  = iftExtractSamples(Z, IFT_TRAIN);
  iftSVM *svm = iftCreateSVM(IFT_RBF, IFT_OVO, C, gamma);
  svm->params->probability = 1;
  printf("Training with split dataset...\n");
  iftSVMTrain(svm, Z1);
  iftDataSet *Z2  = iftExtractSamples(Z, IFT_TEST);
  printf("Classifying split dataset...\n");
  iftSVMClassify(svm, Z2, IFT_TEST);

  printf("Calculating error rates...\n");
  // Calculate P( label, bin ) and P( misclassification | label, bin )
  for (int s = 0; s < Z2->nsamples; ++s) {
    int bin = iftMin(floor(Z2->sample[s].weight * ((float)nBins)), nBins-1);
    int label = Z2->sample[s].label;

    iftMatrixElem(binSampleCount, label, bin) += 1.0f; 
    if (label != Z2->sample[s].truelabel)
      iftMatrixElem(binErrorCount, label, bin) += 1.0f; 
  }
  for (int label = 0; label < Z2->nclasses; ++label) {
    for (int bin = 0; bin < nBins; ++bin) {
      iftMatrixElem(binSampleProb, label, bin) = 
        iftMatrix(binSampleCount, label, bin) / ((float)Z2->nsamples);

      if (iftMatrixElem(binSampleCount, label, bin) < IFT_EPSILON)
        iftMatrixElem(binErrorProb, label, bin) = 0.0f;
      else
        iftMatrixElem(binErrorProb, label, bin) = 
          iftMatrixElem(binErrorCount, label, bin) /
          iftMatrixElem(binSampleCount, label, bin);
    }
  }

  // Train SVM with all images
  iftSetStatus(Z, IFT_TRAIN);
  printf("Training with full dataset...\n");
  iftSVMTrain(svm, Z);
  
  // Save results
  iftWriteRawMatrix(tgt, outMxPath);
  iftWriteSVM(svm, outSvmPath);

  // Cleanup
  iftDestroyDataSet(&Z);
  iftDestroyMatrix(&tgt);
  iftDestroyMatrix(&binSampleCount);
  iftDestroyMatrixPointer(&binSampleProb);
  iftDestroyMatrix(&binErrorCount);
  iftDestroyMatrixPointer(&binErrorProb);
  iftDestroySampler(&sampler);
  iftDestroySVM(svm);
  iftDestroyDataSet(&Z1);
  iftDestroyDataSet(&Z2);

  return 0;
}
