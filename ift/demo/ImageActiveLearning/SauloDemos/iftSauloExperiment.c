#include <ift.h>
#include "iftHybridNetwork.h"

int main(int argc, char* argv[])
{
  if (argc < 6) {
    printf("Usage %s <net_model.json> <net_model.weights> <test.csv> <minibatch_size(default=16)> <sample_results.csv>\n", argv[0]);
    return -1;
  }

  iftNeuralNetwork* net = iftLoadVggFromJson(argv[1], argv[2], atol(argv[4])); 
  FILE *sampleResults = fopen(argv[5], "w");
  iftFileSet *fileset = iftLoadFileSetFromDirOrCSV(argv[3], 1, true);
  for (int f = 0; f < fileset->n; ++f)
    fileset->files[f]->status = IFT_TEST;

  timer *t1 = iftTic();
  // Fill batch and classify loop
  iftNeuralLayer *resLayer = net->layers[net->nlayers - 1];
  int nSamples = fileset->n;
  int nClasses = resLayer->out->n / net->miniBatch;
  iftTensor *pred = iftCreateTensor(2, nSamples, nClasses);
  int offset = 0;
  int batchNum = 1;
  while (offset < nSamples) {
    iftClearEmptyImageLayerBatch(net);
    for (int i = 0; i < net->miniBatch; ++i) {
      int sample = i + offset;
      if (sample < nSamples)
        iftVggLoadImgFromPath(net, fileset->files[sample]->path);
    }

    printf("Starting forward for batch %d (%d/%d)\n", batchNum++, offset, nSamples);
    timer *batchT1 = iftTic();
    iftNetForward(net);
    timer *batchT2 = iftToc();
    float batchTime = iftCompTime(batchT1, batchT2);
    printf("Batch forward finished in %fms (%fms per sample)\n", batchTime, batchTime / net->miniBatch);

    // Fill in predictions
    for (int i = 0; i < net->miniBatch; ++i) {
      int sample = i + offset;
      if (sample >= nSamples)
        continue;
      
      for (int label = 0; label < nClasses; ++label) {
        pred->val[sample * pred->accDimension[0] + label] =
          resLayer->out->val[i * resLayer->out->accDimension[0] + label];
      }
    }

    offset += net->miniBatch;
  }
  timer *t2 = iftToc();
  float classificationTime = iftCompTime(t1,t2);
  printf("Full classification in %fs, %fms per sample\n", classificationTime * 0.001f, classificationTime / ((float)nSamples));
  float *labelError = iftAllocFloatArray(pred->dimension[1]);
  float *labelCount = iftAllocFloatArray(pred->dimension[1]);
  for (int c = 0; c < pred->dimension[1]; ++c) {
    labelError[c] = 0.0f;
    labelCount[c] = 0.0f;
  }

  float nErrors = 0.0f;
  for (int sample = 0; sample < fileset->n; ++sample) {
    fprintf(sampleResults,"%s,", fileset->files[sample]->path);
    int label = -1;
    float maxVal = IFT_INFINITY_FLT_NEG;
    for (int col = 0; col < pred->dimension[1]; ++col) {
      fprintf(sampleResults, "%f,", iftTensorElem(pred, sample, col));
      if (maxVal < iftTensorElem(pred, sample, col)) {
        label = col+1;
	maxVal = iftTensorElem(pred, sample, col);
      }
    }

    assert(label > 0);
    int trueLabel = fileset->files[sample]->label;
    fprintf(sampleResults, "%d\n", trueLabel);
    labelCount[trueLabel - 1] += 1.0f;
    if (label != trueLabel) {
      nErrors += 1.0f;
      labelError[trueLabel-1] += 1.0f;
    }
  }

  printf("Overall accuracy = %f\n", (((float) fileset->n) - nErrors) / ((float) fileset->n));
  for (int c = 0; c < pred->dimension[1]; ++c)
    printf("Class %d accuracy = %f/%f\n", c, labelCount[c] - labelError[c], labelCount[c]);

  fclose(sampleResults);
  iftDestroyNeuralNetwork(&net);

  return 0;
}

