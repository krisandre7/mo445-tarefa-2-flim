#include "ift.h"

int main(int argc, char *argv[])
{
  if (argc < 7) {
    printf("Usage: %s <train_dataset.zip> <val_dataset.zip> <nBins> <svm_params.txt> <error_data.npy> <svm.zip>\n", argv[0]);
    return -1;
  }

  iftDataSet *Ztrain = iftReadDataSet(argv[1]);
  iftDataSet *Zval = iftReadDataSet(argv[2]);
  int nBins = atoi(argv[3]);
  FILE *fp = fopen(argv[4], "r");
  char *outMxPath = argv[5];
  char *outSvmPath = argv[6];

  // read SVM parameters
  float C, gamma;
  fscanf(fp, "%f %f\n", &C, &gamma);
  fclose(fp);


  // Target results
  iftMatrix *tgt = iftCreateMatrix(Ztrain->nclasses, nBins * 2);
  iftMatrix *binSampleCount = iftCreateMatrix(Ztrain->nclasses, nBins); 
  iftMatrix *binSampleProb = iftCreateMatrixPointer(tgt->val, Ztrain->nclasses, nBins);
  iftMatrix *binErrorCount = iftCreateMatrix(Ztrain->nclasses, nBins);
  iftMatrix *binErrorProb = iftCreateMatrixPointer(&(tgt->val[binSampleProb->n]), Ztrain->nclasses, nBins);

  // Train/Classify to obtain error probabilities
  iftSetStatus(Ztrain, IFT_TRAIN);
  iftSetStatus(Zval, IFT_TEST);
  iftSVM *svm = iftCreateSVM(IFT_RBF, IFT_OVO, C, gamma);
  svm->params->probability = 1;
  printf("Training SVM...\n");
  iftSVMTrain(svm, Ztrain);
  printf("Classifying validation dataset...\n");
  iftSVMClassify(svm, Zval, IFT_TEST);

  printf("Calculating error rates...\n");
  // Calculate P( label, bin ) and P( misclassification | label, bin )
  for (int s = 0; s < Zval->nsamples; ++s) {
    int bin = iftMin(floor(Zval->sample[s].weight * ((float)nBins)), nBins-1);
    int label = Zval->sample[s].label - 1;

    iftMatrixElem(binSampleCount, label, bin) += 1.0f; 
    if (label != Zval->sample[s].truelabel - 1)
      iftMatrixElem(binErrorCount, label, bin) += 1.0f; 
  }
  for (int label = 0; label < Zval->nclasses; ++label) {
    for (int bin = 0; bin < nBins; ++bin) {
      iftMatrixElem(binSampleProb, label, bin) =
        iftMatrixElem(binSampleCount, label, bin) / ((float) Zval->nsamples);

      if (iftMatrixElem(binSampleCount, label, bin) < IFT_EPSILON)
        iftMatrixElem(binErrorProb, label, bin) = 0.0f;
      else
        iftMatrixElem(binErrorProb, label, bin) = 
          iftMatrixElem(binErrorCount, label, bin) /
          iftMatrixElem(binSampleCount, label, bin);
    }
  }

  // Save results
  iftWriteMatrix(tgt, outMxPath);
  iftWriteSVM(svm, outSvmPath);

  iftPrintMatrix(tgt);

  // Cleanup
  iftDestroyMatrix(&tgt);
  iftDestroyMatrix(&binSampleCount);
  iftDestroyMatrixPointer(&binSampleProb);
  iftDestroyMatrix(&binErrorCount);
  iftDestroyMatrixPointer(&binErrorProb);
  iftDestroySVM(svm);
  iftDestroyDataSet(&Ztrain);
  iftDestroyDataSet(&Zval);

  return 0;
}
