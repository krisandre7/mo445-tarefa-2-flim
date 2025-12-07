#include <ift.h>
#include "iftHybridNetwork.h"

int main(int argc, char* argv[])
{
  // Not an actual program, just a dummy example to show intended usage of hybrid scheme
  return -1;

  if (argc < 5) {
    printf("Usage %s <trained svm> <cnn arch> <trained cnn weights> <error rates>\n", argv[0]);
    return -1;
  }
  
  iftSVM *svm = iftReadSVM(argv[1]);
  int miniBatchSize = 1;
  iftNeuralNetwork *net = iftLoadVggFromJson(argv[2], argv[3], miniBatchSize);
  iftMatrix *pMatrix = iftReadRawMatrix(argv[4]);
  int nFeats = 10; // Arbitrary example
  float * feats = iftAllocFloatArray(nFeats);

  // First step, classification by SVM
  int svmLabel;
  float svmProb;
  iftSVMClassifyOVO_ProbabilitySingleSample(svm, feats, nFeats, &svmLabel, &svmProb);
  printf("SVM predicted label %d with prob %f\n", svmLabel, svmProb);

  if (iftHybridDecision(svmLabel, svmProb, pMatrix)) {
    // Operation assuming minibatch = 1
    int *R, *G, *B;
    int emptyBatchSlots = iftVggLoadImgFromBuffer(net, R, G, B); 
    (void) emptyBatchSlots; // Avoid warning

    iftNetForward(net);
    iftNeuralLayer *lastLayer = net->layers[net->nlayers - 1];
    int nClasses = lastLayer->out->n;

    float maxVal = IFT_INFINITY_FLT_NEG;
    int netLabel = -1;
    for (int i = 0; i < nClasses; ++i) {
      if (maxVal < lastLayer->out->val[i]) {
        netLabel = i + 1;
	maxVal = lastLayer->out->val[i];
      }
    } 
    printf("Network predicted label %d\n", netLabel);
  }

  // TODO add relevant destroy functions

  return 0;
}
