#include <ift.h>
#include "iftHybridNetwork.h"


int main(int argc, char* argv[])
{
  if (argc < 8) {
    printf("Usage %s <svm.zip> <minibatch_size> <ift_arch.json> <ift_arch.weights> <error_data.npy> <test_dataset.zip> <perc_network = [0,1]>\n", argv[0]);
    return -1;
  }
  
  iftSVM *svm = iftReadSVM(argv[1]);
  int miniBatchSize = atol(argv[2]);
  iftNeuralNetwork *net = iftLoadVggFromJson(argv[3], argv[4], miniBatchSize);
  iftMatrix *errorData = iftReadMatrix(argv[5]);
  iftDataSet *Z = iftReadDataSet(argv[6]);
  iftMatrix *pMatrix = iftGetHybridBinProbabilities(errorData, atof(argv[7]));
  //iftPrintMatrix(pMatrix);
  //iftPrintMatrix(errorData);

  iftFileSet *fileset = Z->ref_data;
  iftSetStatus(Z, IFT_TEST);

  int netCount = 0;
  int emptyBatchSlots = net->miniBatch;
  int *batchSampleIdx = iftAllocIntArray(net->miniBatch);
  int *batchSamplePrediction = iftAllocIntArray(net->miniBatch);

  for (int sample = 0; sample < Z->nsamples; ++sample) {
    int svmLabel;
    float svmProb;

    iftSVMClassifyOVO_ProbabilitySingleSample(svm, Z->sample[sample].feat,
        Z->nfeats, &svmLabel, &svmProb);
    Z->sample[sample].label = svmLabel;

    if (iftHybridDecision(svmLabel, svmProb, pMatrix)) {
      netCount += 1;
      const char *imgPath = fileset->files[sample]->path;
      emptyBatchSlots = iftVggLoadImgFromPath(net, imgPath);
      batchSampleIdx[net->miniBatch - emptyBatchSlots] = sample;

      if (emptyBatchSlots == 0 || net->miniBatch == 1) {
        printf("Filled batch at sample %d, running network:\n", sample);
        iftNetForward(net);
        iftClearEmptyImageLayerBatch(net);
        emptyBatchSlots = net->miniBatch;
        iftGetNetworkPredictions(net, batchSamplePrediction);
        for (int i = 0; i < net->miniBatch; ++i)
          Z->sample[batchSampleIdx[i]].label = batchSamplePrediction[i];
      }
    }
  }

  // Run remaining incomplete batch if necessary
  if (emptyBatchSlots != net->miniBatch) {
    iftNetForward(net);
    iftClearEmptyImageLayerBatch(net);
    iftGetNetworkPredictions(net, batchSamplePrediction);
    for (int i = 0; i < net->miniBatch - emptyBatchSlots; ++i)
      Z->sample[batchSampleIdx[i]].label = batchSamplePrediction[i];
    emptyBatchSlots = net->miniBatch;
  }

  // Results
  printf("Finished hybrid classification with %d samples, %d(%f%%) with CNN.\n",
      Z->nsamples, netCount, 100.0f * (((float)netCount)/((float)Z->nsamples)));
  printf("Kappa = %f\n", iftCohenKappaScore(Z));
  
  int nErrors = 0;
  int *nErrorsPerClass = iftAllocIntArray(Z->nclasses);
  int *nSamplesPerClass = iftAllocIntArray(Z->nsamples);
  for (int sample = 0; sample < Z->nsamples; ++sample) {
    int label = Z->sample[sample].label;
    int trueLabel = Z->sample[sample].truelabel;
    nSamplesPerClass[trueLabel - 1] += 1;
    if (label != trueLabel) {
      nErrorsPerClass[trueLabel - 1] += 1;
      nErrors += 1;
    }
  }

  float acc = 1.0f - (((float) nErrors) / ((float) Z->nsamples));
  printf("Overall accuracy = %f (%d/%d)\n", acc, nErrors, Z->nsamples);
  for (int i = 0; i < Z->nclasses; ++i) {
    acc = 1.0f - (((float) nErrorsPerClass[i]) / ((float) nSamplesPerClass[i]));
    printf("Class %d accuracy = %f (%d/%d)\n", i, acc, nErrorsPerClass[i], nSamplesPerClass[i]);
  }

  // Clean-up
  iftDestroySVM(svm);
  iftDestroyNeuralNetwork(&net);
  iftDestroyMatrix(&errorData);
  iftDestroyDataSet(&Z);
  iftDestroyMatrix(&pMatrix);
  free(nErrorsPerClass);
  free(nSamplesPerClass);

  return 0;
}
