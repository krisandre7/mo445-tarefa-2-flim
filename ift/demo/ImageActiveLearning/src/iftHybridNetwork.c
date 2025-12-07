#include "iftHybridNetwork.h"

iftNeuralNetwork* iftLoadVggFromJson(const char *json_path, const char *weights_path, int minibatch_size)
{
  // Sanitize input
  if (json_path == NULL)
    iftError("The Vgg architecture Json file is required.", "iftLoadVggFromJson");
  if (json_path == NULL)
    iftError("The Vgg architecture weights file is required.", "iftLoadVggFromJson");
  if (minibatch_size <= 0)
    minibatch_size = 1;

  iftNeuralNetwork *net = iftCreateWeightedNeuralNetworkFromJson(json_path, weights_path, NULL, minibatch_size, true);

  return net;
}

int iftVggLoadImgFromPath(iftNeuralNetwork* net, const char *img_path)
{
  iftNeuralLayer* layer = net->layers[0];
  assert(layer->type == IFT_IMAGE_LAYER);
  assert(layer->idxCounter < net->miniBatch);

  int i = layer->idxCounter;
  iftImage* img = iftReadImageByExt(img_path);
  assert(img->xsize == layer->xdata);
  assert(img->ysize == layer->ydata);
  assert(img->zsize == layer->zdata);
  assert(iftIsColorImage(img));
  assert(layer->ninput == 3);

  iftImage* channel0 = iftImageRed(img);
  iftImage* channel1 = iftImageGreen(img);
  iftImage* channel2 = iftImageBlue(img);

  for (int x = 0; x < layer->xdata; ++x) {
    for (int y = 0; y < layer->ydata; ++y) {
      for (int z = 0; z < layer->zdata; ++z) {
        iftTensorElem(layer->out, i, 0, x, y, z) = (((float) iftImgVal(channel0, x, y, z))/255.0f) - 103.939f;
        iftTensorElem(layer->out, i, 1, x, y, z) = (((float) iftImgVal(channel1, x, y, z))/255.0f) - 116.779f;
        iftTensorElem(layer->out, i, 2, x, y, z) = (((float) iftImgVal(channel2, x, y, z))/255.0f) - 123.68f;
      }
    }
  }

  iftDestroyImage(&channel0);
  iftDestroyImage(&channel1);
  iftDestroyImage(&channel2);

  if (net->miniBatch > 1) {
    layer->idxCounter++;
    return net->miniBatch - layer->idxCounter;
  } else {
    // Simplify operation without batches
    return 1;
  }
}

int iftVggLoadImgFromBuffer(iftNeuralNetwork *net, int *R, int *G, int *B)
{
  iftNeuralLayer* layer = net->layers[0];
  assert(layer->type == IFT_IMAGE_LAYER);
  assert(layer->ninput == 3);
  assert(layer->idxCounter < net->miniBatch);

  int i = layer->idxCounter;
  int idx = 0;
  for (int z = 0; z < layer->zdata; ++z) {
    for (int y = 0; y < layer->ydata; ++y) {
      for (int x = 0; x < layer->xdata; ++x) {
        iftTensorElem(layer->out, i, 0, x, y, z) = (((float) R[idx])/255.0f) - 103.939f;
        iftTensorElem(layer->out, i, 1, x, y, z) = (((float) G[idx])/255.0f) - 116.779f;
        iftTensorElem(layer->out, i, 2, x, y, z) = (((float) B[idx])/255.0f) - 123.68f;
        idx++;
      }
    }
  }

  if (net->miniBatch > 1) {
    layer->idxCounter++;
    return net->miniBatch - layer->idxCounter;
  } else {
    // Simplify operation without batches
    return 1;
  }
}

void iftSVMClassifyOVO_ProbabilitySingleSample(  iftSVM *svm, float *feats, int nFeats, int *outLabel, float *outProb)
{
  //assert(nFeats == svm->model[0]->l); // TODO check if that's the correct parameter 
  int nClasses = svm->model[0]->nr_class;
  int precomp = svm->params->kernel_type == PRECOMPUTED;
  double *prob_estimates = (double *) malloc(nClasses*sizeof(double));

  if (svm->nmodels == 0)
    iftError("SVM not trained yet", "iftSVMClassifyOVO_ProbabilitySingleSample");

  if (svm->truelabel != NULL)
    iftError("This is not a p-SVM", "iftSVMClassifyOVO_ProbabilitySingleSample");

  // temporary buffer to get predictions from libsvm
  svmNode *x = (svmNode *) iftAlloc(precomp + nFeats + 1, sizeof(svmNode));
  if (precomp) {
    x[0].index = 0;
    x[0].value = 1.0f;
  }
  for (int f = 0; f < nFeats; ++f) {
    x[precomp + f].index = f + 1;
    x[precomp + f].value = (double) feats[f];
  }
  x[precomp + nFeats].index = -1;
  x[precomp + nFeats].value = 0.0;

  *outLabel = (int) svm_predict_probability(svm->model[0], x, prob_estimates);
  *outProb = prob_estimates[(*outLabel) - 1];

  free(prob_estimates);
  iftFree(x);
}

iftMatrix * iftGetHybridBinProbabilities(iftMatrix *errorData, float netFraction)
{
  int nLabels = errorData->ncols;
  int nBins = errorData->nrows / 2;
  iftMatrix *res = iftCreateMatrix(nLabels, nBins);
  iftMatrix *binSampleProb = iftCreateMatrixPointer(errorData->val, nLabels, nBins);
  iftMatrix *binErrorProb = iftCreateMatrixPointer(errorData->val, nLabels, nBins);

  // Calculate error mean
  float weightedErrorMean = 0.0f;
  for (int label = 0; label < nLabels; ++label)
    for (int bin = 0; bin < nBins; ++bin)
      weightedErrorMean += iftMatrixElem(binErrorProb, label, bin) * iftMatrixElem(binSampleProb, label, bin); 
  weightedErrorMean /= ((float)res->n);

  // Fill result matrix with effective probability
  for (int label = 0; label < nLabels; ++label) {
    for (int bin = 0; bin < nBins; ++bin) {
      float weight = iftMatrixElem(binErrorProb, label, bin) / weightedErrorMean;
      float prob = iftMin(netFraction * weight, 1.0f);
      iftMatrixElem(res, label, bin) = prob;
    }
  }

  iftDestroyMatrixPointer(&binSampleProb);
  iftDestroyMatrixPointer(&binErrorProb);

  return res;
}

bool iftHybridDecision(int svmLabel, float svmProb, iftMatrix *binProb)
{
  int nBins = binProb->nrows;
  int bin = iftMin(floor(svmProb * ((float)nBins)), nBins-1);
  return iftRandomUniform(0.0f, 1.0f) < iftMatrixElem(binProb, svmLabel, bin);
}

