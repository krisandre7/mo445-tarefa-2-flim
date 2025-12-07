#include <ift.h>
#include "iftHybridNetwork.h"

void iftGetMiniBatchPredictions(iftNeuralNetwork *net, int *batchSampleIdx, int nBatchSamples, iftDataSet *Z)
{
  assert(net != NULL);
  assert(batchSampleIdx != NULL);
  assert(nBatchSamples >= 0 && nBatchSamples <= net->miniBatch);
  assert(Z != NULL);

  iftNeuralLayer *predLayer = net->layers[net->nlayers - 1];
  int nClasses = predLayer->out->n / net->miniBatch;
  assert(nClasses == Z->nclasses);

  for (int i = 0; i < nBatchSamples; ++i) {
    float maxVal = IFT_INFINITY_FLT_NEG;
    for (int label = 0; label < nClasses; ++label) {
      float predWeight = predLayer->out->val[i * predLayer->out->accDimension[0] + label];
      if (predWeight > maxVal) {
        maxVal = predWeight;
        int sample = batchSampleIdx[i];
        Z->sample[sample].label = label + 1;
      }
    }
  }
}

int main(int argc, char* argv[])
{
  static   int miniBatchSize = 16;

  if (argc < 7) {
    printf("Usage %s <trained svm> <cnn arch> <trained cnn weights> <error_data> <dataset> <perc_network = [0,1]>\n", argv[0]);
    return -1;
  }
  
  iftSVM *svm = iftReadSVM(argv[1]);
  iftNeuralNetwork *net = iftLoadVggFromJson(argv[2], argv[3], miniBatchSize);
  iftMatrix *errorData = iftReadRawMatrix(argv[4]);
  iftDataSet *Z = iftReadOPFDataSet(argv[5]);
  iftMatrix *pMatrix = iftGetHybridBinProbabilities(errorData, atof(argv[6]));
  iftFileSet *fileset = iftLoadFileSetFromDirOrCSV((char *) Z->ref_data, 1, true);
  iftSetStatus(Z, IFT_TEST);

  int netCount = 0;
  int emptyBatchSlots = net->miniBatch;
  int *batchSampleIdx = iftAllocIntArray(net->miniBatch);

  for (int sample = 0; sample < Z->nsamples; ++sample) {
    int svmLabel;
    float svmProb;

    iftSVMClassifyOVO_ProbabilitySingleSample(svm, Z->sample[sample].feat,
        Z->nfeats, &svmLabel, &svmProb);
    Z->sample[sample].label = svmLabel;

    if (iftHybridDecision(svmLabel, svmProb, pMatrix)) {
      netCount += 1;
      const char *imgPath = fileset->files[Z->sample[sample].id]->path;
      emptyBatchSlots = iftVggLoadImgFromPath(net, imgPath);
      batchSampleIdx[net->miniBatch - emptyBatchSlots] = sample;

      if (!emptyBatchSlots || net->miniBatch == 1) {
        iftNetForward(net);
        iftClearEmptyImageLayerBatch(net);
        emptyBatchSlots = net->miniBatch;
        iftGetMiniBatchPredictions(net, batchSampleIdx, net->miniBatch, Z);
      }
    }
  }

  // Run remaining incomplete batch if necessary
  iftGetMiniBatchPredictions(net, batchSampleIdx, net->miniBatch - emptyBatchSlots, Z);

  // Results
  printf("Finished hybrid classification with %d samples, %d(%f%%) with CNN.\n",
      Z->nsamples, netCount, ((float)netCount)/((float)Z->nsamples));
  printf("Kappa = %f\n", iftCohenKappaScore(Z));
  bool printAccByClass = true;
  float acc = iftOPFAccuracy(Z, printAccByClass);
  printf("Overall accuracy = %f\n", acc);

  // Clean-up
  iftDestroySVM(svm);
  iftDestroyNeuralNetwork(&net);
  iftDestroyMatrix(&errorData);
  iftDestroyDataSet(&Z);
  iftDestroyMatrix(&pMatrix);
  iftDestroyFileSet(&fileset);

  return 0;
}
