#include "ift.h"
#include "iftHybridNetwork.h"

int main(int argc, char *argv[])
{
  if (argc < 3) {
    printf("Usage %s <svm.zip> <test_dataset.zip>\n", argv[0]);
    return -1;
  }

  iftSVM *svm = iftReadSVM(argv[1]);
  iftDataSet *Z = iftReadDataSet(argv[2]);

  int *classCount = iftAllocIntArray(Z->nclasses);
  int *errorCount = iftAllocIntArray(Z->nclasses);
  int totalErrors = 0; 
  iftSetStatus(Z, IFT_TEST);

  for (int sample = 0; sample < Z->nsamples; ++sample) {
    int svmLabel;
    float svmProb;

    // Using single sample function to make sure it works
    iftSVMClassifyOVO_ProbabilitySingleSample(svm, Z->sample[sample].feat,
        Z->nfeats, &svmLabel, &svmProb);

    int trueLabel = Z->sample[sample].truelabel;
    Z->sample[sample].label = svmLabel;
    classCount[trueLabel-1] += 1;
    if (svmLabel != trueLabel) {
      errorCount[trueLabel-1] += 1;
      totalErrors += 1;
    }
  }

  printf("Kappa = %f\n", iftCohenKappaScore(Z));
  float acc = 1.0f - (((float) totalErrors) / ((float) Z->nsamples));
  printf("Overall accuracy = %f (%d/%d)\n", acc, totalErrors, Z->nsamples);
  for (int i = 0; i < Z->nclasses; ++i) {
    acc = 1.0f - (((float) errorCount[i]) / ((float) classCount[i]));
    printf("Class %d accuracy = %f (%d/%d)\n", i, acc, errorCount[i], classCount[i]);
  }

  iftDestroySVM(svm);
  iftDestroyDataSet(&Z);

  return 0;
}
