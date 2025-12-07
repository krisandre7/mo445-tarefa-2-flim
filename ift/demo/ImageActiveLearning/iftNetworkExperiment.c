#include <ift.h>
#include "iftHybridNetwork.h"

int main(int argc, char* argv[])
{
  if (argc < 4) {
    printf("Usage %s <model.json> <weights> <test_csv>\n", argv[0]);
    return -1;
  }
  
  iftNeuralNetwork* net = iftLoadVggFromJson(argv[1], argv[2], 1); 
  iftFileSet *fileset = iftLoadFileSetFromDirOrCSV(argv[3], 1, true);
  iftNeuralLayer *lastLayer = net->layers[net->nlayers - 1];
  int nClasses = lastLayer->out->n;
  float *labelError = iftAllocFloatArray(nClasses);
  float *labelCount = iftAllocFloatArray(nClasses);
  for (int c = 0; c < nClasses; ++c) {
    labelError[c] = 0.0f;
    labelCount[c] = 0.0f;
  }

  float nErrors = 0.0f;
  for (int sample = 0; sample < fileset->n; ++sample) {
    int label = -1;
    iftVggLoadImgFromPath(net, fileset->files[sample]->path);

    fprintf(stderr, "Starting forward with sample %d/%d: %s\n", sample+1, fileset->n, fileset->files[sample]->path);
    timer *t1 = iftTic();
    iftNetForward(net);
    timer *t2 = iftToc();
    fprintf(stderr, "Network forward time: %fms\n", iftCompTime(t1, t2));

    // DEBUG: dump of output layer
    /*printf("sample %d:", sample);
    for (int i = 0; i < nClasses; ++i)
      printf(" %f", lastLayer->out->val[i]);
    printf("\n");*/

    float maxVal = IFT_INFINITY_FLT_NEG;
    for (int i = 0; i < nClasses; ++i) {
      if (maxVal < lastLayer->out->val[i]) {
        label = i + 1;
	maxVal = lastLayer->out->val[i];
      }
    } 

    assert(label > 0);
    int trueLabel = fileset->files[sample]->label;
    labelCount[trueLabel - 1] += 1.0f;
    fprintf(stderr, "Got %d expected %d\n", label, trueLabel);
    if (label != trueLabel) {
      nErrors += 1.0f;
      labelError[trueLabel-1] += 1.0f;
    }
  }

  printf("Overall accuracy = %f\n", (((float) fileset->n) - nErrors) / ((float) fileset->n));
  for (int c = 0; c < nClasses; ++c)
    printf("Class %d errors = %f/%f\n", c+1, labelError[c], labelCount[c]);

  iftDestroyNeuralNetwork(&net);

  return 0;
}
