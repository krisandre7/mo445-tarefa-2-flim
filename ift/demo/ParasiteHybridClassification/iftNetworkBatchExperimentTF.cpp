#include <ift.h>
#include "Model.h"
#include "Tensor.h"
#include "iftHybridNetwork.h"
#include "iftHybridNetworkTF.h"

int main(int argc, char* argv[])
{
  if (argc < 4) {
    printf("Usage %s <saved_model.pb> <test.csv> <minibatch_size(default=16)>\n", argv[0]);
    return -1;
  }

  Model model(argv[1]);
  iftFileSet *fileset = iftLoadFileSetFromDirOrCSV(argv[2], 1, true);
  int batchSize = atol(argv[3]);
  for (int f = 0; f < fileset->n; ++f)
    fileset->files[f]->status = IFT_TEST;

  // Network info
  std::vector<float> data(PHN::imgLen * batchSize);
  Tensor net_input{model, "x"};
  Tensor net_output{model, "Identity"};

  timer *t1 = iftTic();
  // Fill batch and classify loop
  int nSamples = fileset->n;
  int nClasses = net_output.get_shape()[1];
  iftTensor *pred = iftCreateTensor(2, nSamples, nClasses);
  int offset = 0;
  int batchNum = 1;
  while (offset < nSamples) {
    int batchCounter = 0;
    for (int i = 0; i < batchSize; ++i) {
      int sample = i + offset;
      if (sample < nSamples)
        iftVggLoadImgFromPath_TF(data, batchCounter++, fileset->files[sample]->path);
    }
    net_input.set_data(data);

    printf("Starting forward for batch %d (%d/%d)\n", batchNum++, offset, nSamples);
    timer *batchT1 = iftTic();
    model.run(net_input, net_output);
    timer *batchT2 = iftToc();
    float batchTime = iftCompTime(batchT1, batchT2);
    printf("Batch forward finished in %fms (%fms per sample)\n", batchTime, batchTime / batchCounter);

    // Fill in predictions
    for (int i = 0; i < batchCounter; ++i) {
      int sample = i + offset;
      if (sample >= nSamples)
        continue;
      
      for (int label = 0; label < nClasses; ++label) {
        pred->val[sample * pred->accDimension[0] + label] =
          net_output.get_data<float>()[i * nClasses + label];
      }
    }

    offset += batchSize;
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
    int label = -1;
    float maxVal = IFT_INFINITY_FLT_NEG;
    for (int col = 0; col < pred->dimension[1]; ++col) {
      if (maxVal < iftTensorElem(pred, sample, col)) {
        label = col+1;
	maxVal = iftTensorElem(pred, sample, col);
      }
    }

    assert(label > 0);
    int trueLabel = fileset->files[sample]->label;
    labelCount[trueLabel - 1] += 1.0f;
    if (label != trueLabel) {
      nErrors += 1.0f;
      labelError[trueLabel-1] += 1.0f;
    }
  }

  printf("Overall accuracy = %f\n", (((float) fileset->n) - nErrors) / ((float) fileset->n));
  for (int c = 0; c < pred->dimension[1]; ++c)
    printf("Class %d accuracy = %f/%f\n", c, labelCount[c] - labelError[c], labelCount[c]);

  return 0;
}

