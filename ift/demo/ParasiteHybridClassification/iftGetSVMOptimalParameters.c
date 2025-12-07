#include "ift.h"

int main(int argc, char* argv[])
{
  if (argc < 3) {
    printf("Usage: %s <train_dataset.zip> <svm_params.txt>\n", argv[0]);
    exit(-1);
  }

  // Split training set
  iftDataSet *train = iftReadDataSet(argv[1]);
  char *outPath = argv[2];
  int nTrainSamples = (int) (((float) train->nsamples) * 0.5f);
  iftSampler* sampler = iftRandomSubsampling(train->nsamples, 1, nTrainSamples);
  iftDataSetSampling(train, sampler, 0);
  iftDataSet *Z1 = iftExtractSamples(train, IFT_TRAIN);
  iftDataSet *Z2 = iftExtractSamples(train, IFT_TEST);

  // p-SVM init
  iftSVM *svm= iftCreateSVM(IFT_RBF, IFT_OVO, 0.0f, 0.0f);
  svm->params->probability = 1;

  float bestExpC = -1.0f;
  float bestExpGamma = -1.0f;
  float bestAcc = 0.0f;

  printf("Starting loose grid search:\n");
  for (float expC = -5.0f; expC < 16.0f; expC += 2.0f) {
    svm->params->C = pow(2.0f, expC);
    for (float expG = -15.0f; expG <= 4.0f; expG += 2.0f) {
      svm->params->gamma = pow(2.0f, expG);

      iftSVMTrain(svm, Z1);
      iftSetStatus(Z2, IFT_TEST);
      iftSVMClassify(svm, Z2, IFT_TEST);
      float acc = iftCohenKappaScore(Z2);

      if (acc > bestAcc){
        bestExpGamma = expG;
        bestExpC = expC;
        bestAcc = acc;
      }
      printf("Tested: C = %f, gamma = %f, Acc = %f\n",
          svm->params->C, svm->params->gamma, acc);
    }
  }

  printf("Starting finer grid search:\n");
  for (float expC = bestExpC - 2.0f; expC < bestExpC + 2.1f; expC += 0.25f) {
    svm->params->C = pow(2.0f, expC);
    for (float expG = bestExpGamma - 2.0f; expG <= bestExpGamma + 2.1f; expG += 0.25f) {
      svm->params->gamma = pow(2.0f, expG);

      iftSVMTrain(svm, Z1);
      iftSetStatus(Z2, IFT_TEST);
      iftSVMClassify(svm, Z2, IFT_TEST);
      float acc = iftCohenKappaScore(Z2);

      if (acc > bestAcc){
        bestExpGamma = expG;
        bestExpC = expC;
        bestAcc = acc;
      }
      printf("Tested: C = %f, gamma = %f, Acc = %f\n",
          svm->params->C, svm->params->gamma, acc);
    }
  }

  printf("Best accuracy = %f\n", bestAcc);
  printf("Best C = %f (2^%f)\n", pow(2.0f, bestExpC), bestExpC);
  printf("Best gamma = %f (2^%f)\n", pow(2.0f, bestExpGamma), bestExpGamma);

  FILE *fp = fopen(outPath, "w");
  fprintf(fp, "%f %f\n", pow(2, bestExpC), pow(2, bestExpGamma));
  fclose(fp);

  iftDestroyDataSet(&train);
  iftDestroySampler(&sampler);
  iftDestroyDataSet(&Z1);
  iftDestroyDataSet(&Z2);
  iftDestroySVM(svm);

  return 0;
}

