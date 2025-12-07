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

int iftVggLoadImg(iftNeuralNetwork* net, iftImage *img)
{
  assert(net != NULL);
  assert(img != NULL);
  assert(iftIsColorImage(img));
  assert(img->xsize == net->layers[0]->layer->xdata);
  assert(img->ysize == net->layers[0]->layer->ydata);
  assert(img->zsize == net->layers[0]->layer->zdata);
  assert(net->layers[0]->ninput == 3);

  iftImage* R = iftImageRed(img);
  iftImage* G = iftImageGreen(img);
  iftImage* B = iftImageBlue(img);

  int res = iftVggLoadImgFromBuffer(net, R->val, G->val, B->val, iftImageDepth(img));

  iftDestroyImage(&R);
  iftDestroyImage(&G);
  iftDestroyImage(&B);

  return res;
}

int iftVggLoadImgFromPath(iftNeuralNetwork* net, const char *img_path)
{
  assert(net != NULL);
  assert(img_path != NULL);

  iftImage* img = iftReadImageByExt(img_path);
  int res = iftVggLoadImg(net, img);
  iftDestroyImage(&img);

  return res;
}

int iftVggLoadImgFromBuffer(iftNeuralNetwork *net, int *R, int *G, int *B, int depth)
{
  iftNeuralLayer* layer = net->layers[0];
  assert(layer->type == IFT_IMAGE_LAYER);
  assert(layer->ninput == 3);
  assert(layer->idxCounter < net->miniBatch);
  assert(depth >= 8);

  float depthCorrection = (float) (1 << (depth - 8));

  // Should match kerasVggUtils.py:LoadDatasetWithVggPreProc
  // RGB->BGR, 8-bit depth conversion and subtract ImageNet mean (Initialization for Vgg16)
  int i = layer->idxCounter;
  int idx = 0;
  for (int z = 0; z < layer->zdata; ++z) {
    for (int y = 0; y < layer->ydata; ++y) {
      for (int x = 0; x < layer->xdata; ++x) {
        iftTensorElem(layer->out, i, 0, x, y, z) =
          ((float) B[idx]) / depthCorrection - 103.939f;
        iftTensorElem(layer->out, i, 1, x, y, z) =
          ((float) G[idx]) / depthCorrection - 116.779f;
        iftTensorElem(layer->out, i, 2, x, y, z) =
          ((float) R[idx]) / depthCorrection - 123.68f;
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
  assert(nFeats == svm->model[0]->l);
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
  iftMatrix *binSampleProb = iftCreateMatrixPointer(
      errorData->val, nLabels, nBins);
  iftMatrix *binErrorProb = iftCreateMatrixPointer(
      &(errorData->val[binSampleProb->n]), nLabels, nBins);

  // Mark (label, bin) pairs with probability higher than 1 
  iftMatrix *saturationMask = iftCreateMatrix(nLabels, nBins);
  int nSaturated = 0;
  float saturatedTotal;

  do {
    // Calculate error mean
    float weightedErrorMean = 0.0f;
    for (int label = 0; label < nLabels; ++label)
      for (int bin = 0; bin < nBins; ++bin)
        if (iftMatrixElem(saturationMask, label, bin) < IFT_EPSILON)
          weightedErrorMean += (iftMatrixElem(binErrorProb, label, bin) *
              iftMatrixElem(binSampleProb, label, bin)); 
    // No division as sum of probabilities = 1

    // Fill result matrix with effective probability
    saturatedTotal = 0.0f;
    for (int label = 0; label < nLabels; ++label) {
      for (int bin = 0; bin < nBins; ++bin) {
        if (iftMatrixElem(saturationMask, label, bin) < IFT_EPSILON) {
          float weight = weightedErrorMean > IFT_EPSILON ? 
            iftMatrixElem(binErrorProb, label, bin) / weightedErrorMean : 0.0f;
          float prob = netFraction * weight;
          if (prob > 1.0f) {
            nSaturated += 1;
            saturatedTotal += iftMatrixElem(binSampleProb, label, bin);
            iftMatrixElem(saturationMask, label, bin) = 1.0f;
            iftMatrixElem(res, label, bin) = 1.0f;
          } else {
            iftMatrixElem(res, label, bin) = prob;
          }
        }
      }
    }

    netFraction -= saturatedTotal;
    assert(netFraction >= 0.0f);
  } while (saturatedTotal > 0.0f);

  iftDestroyMatrixPointer(&binSampleProb);
  iftDestroyMatrixPointer(&binErrorProb);
  iftDestroyMatrix(&saturationMask);

  return res;
}

bool iftHybridDecision(int svmLabel, float svmProb, iftMatrix *binProb)
{
  svmLabel -= 1; // 1 to 0 indexed
  int nBins = binProb->nrows;
  int bin = iftMin(floor(svmProb * ((float)nBins)), nBins-1);
  return iftRandomUniform(0.0f, 1.0f) <= iftMatrixElem(binProb, svmLabel, bin);
}

iftImage * iftHybridNetworkPreProcImg(iftImage *origImg, iftImage *labelImg)
{
    int SIZE = 200;
  assert(origImg != NULL);
  assert(label != NULL);
  assert(origImg->n == labelImg->n);
  assert(origImg->xsize == labelImg->xsize);
  assert(origImg->ysize == labelImg->ysize);
  assert(origImg->zsize == labelImg->zsize);

  iftImage *aux;
  iftVoxel pos;

  // --- Crop
  iftBoundingBox bb = iftMinBoundingBox(labelImg, &pos);
  iftImage *origROI = iftExtractROI(origImg, bb);
  iftImage *labelROI = iftExtractROI(labelImg, bb);
  aux = origROI;
  origROI = iftMask(origROI, labelROI);
  iftDestroyImage(&aux);

  for (int p = 0; p < labelROI->n; ++p)
    if (labelROI->val[p] != 0)
      labelROI->val[p] = 255;

  // --- Alignment
  iftMatrix *M = iftRotationMatrixToAlignByPrincipalAxes(labelROI);
  iftImage *origAlign = iftTransformImageByMatrix(origROI, M);
  iftImage *labelAlign = iftTransformImageByMatrix(labelROI, M);

  for (int p = 0; p < labelAlign->n; ++p)
    if (labelAlign->val[p] != 0)
      labelAlign->val[p] = 255;
    
  iftDestroyMatrix(&M);
  iftDestroyImage(&origROI);
  iftDestroyImage(&labelROI);

  bb = iftMinBoundingBox(labelAlign, &pos);
  origROI = iftExtractROI(origAlign, bb);
  labelROI = iftExtractROI(labelAlign, bb);

  iftDestroyImage(&origAlign);
  iftDestroyImage(&labelAlign);

  // --- Resize
  float s = (0.9f * SIZE) / iftMax(labelROI->xsize, labelROI->ysize); 
      
  iftImage *origInterp = iftInterp2D(origROI, s ,s);
  iftImage *labelInterp = iftInterp2D(labelROI, s, s);

  for (int p = 0; p < labelInterp->n; ++p)
    if (labelInterp->val[p] != 0)
      labelInterp->val[p] = 255;

  iftDestroyImage(&origROI);
  iftDestroyImage(&labelROI);

  // --- Build final image
  int depth = iftImageDepth(origInterp);
  if (depth != 16)
    iftError("Depth must not be different from 16 bits","");

  iftImage *origRes = iftCreateColorImage(SIZE, SIZE, 1, depth); 
  pos.x = (SIZE - origInterp->xsize) / 2;
  pos.y = (SIZE - origInterp->ysize) / 2;
  pos.z = 0;
  iftInsertROI(origInterp, origRes, pos);
    
  iftImage *labelRes = iftCreateImage(SIZE, SIZE, 1);
  pos.x = (SIZE - labelInterp->xsize) / 2;
  pos.y = (SIZE - labelInterp->ysize) / 2;
  pos.z = 0;
  iftInsertROI(labelInterp, labelRes, pos);
      
  aux = origRes;
  origRes = iftMask(origRes, labelRes);
  iftDestroyImage(&aux);

  iftDestroyImage(&labelRes);
  iftDestroyImage(&origInterp);
  iftDestroyImage(&labelInterp);

  return origRes;
}

void iftUpdateFileSetDir(iftFileSet *fs, char *newDir)
{
  assert(fs != NULL);

  for (int i = 0; i < fs->n; ++i) {
    char *base = iftSplitStringAt(fs->files[i]->path, IFT_SEP_C, -1);
    char buf[256];
    sprintf(buf, "%s%s%s", newDir, IFT_SEP_C, base);
    free(fs->files[i]->path);
    fs->files[i]->path = iftCopyString(buf);
    free(base);
  }
}

void iftGetNetworkPredictions(iftNeuralNetwork *net, int *batchSamplePred)
{
  assert(net != NULL);
  assert(batchSamplePred != NULL);

  iftNeuralLayer *predLayer = net->layers[net->nlayers - 1];
  int nClasses = predLayer->out->n / net->miniBatch;

  for (int i = 0; i < net->miniBatch; ++i) {
    float maxVal = IFT_INFINITY_FLT_NEG;
    for (int label = 0; label < nClasses; ++label) {
      float predWeight = predLayer->out->val[i * predLayer->out->accDimension[0] + label];
      if (predWeight > maxVal) {
        maxVal = predWeight;
        batchSamplePred[i] = label + 1;
      }
    }
  }
}

