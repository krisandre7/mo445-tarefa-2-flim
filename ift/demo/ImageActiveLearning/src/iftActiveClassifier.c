#include <ift.h>
#include "iftActiveClassifier.h"

// ----- Private Interface -----
void iftTrainActiveClassifierOPFSup(iftActiveClassifier * classifier);
void iftTrainActiveClassifierOPFSemi(iftActiveClassifier * classifier);
void iftTrainActiveClassifierLogReg(iftActiveClassifier * classifier); 
iftDataSet * iftExtractUnlabeledSemisupSamples(iftActiveClassifier * classifier);
bool iftIsSemiSupClassifier(iftActiveClassifier * classifier);
void iftResetActiveClassifierDataSets(iftActiveClassifier * classifier);
void iftResetActiveClassifierClassifiers(iftActiveClassifier * classifier);
// TODO move to ift when done
iftMatrix * iftLogRegGetConditionalProb(iftMatrix * featMatrix, iftMatrix * weightMatrix);
iftMatrix * iftLogRegGrad(iftLogReg * lg, iftMatrix * featMatrix, iftMatrix * labelMatrix);
iftList * iftSelectMiniBatchSamples(iftRandomSelector * rs, int batchSize);
void iftFillSampleBatch(iftList * selectedSamples, iftMatrix * batchFeats, iftMatrix * batchLabels, iftMatrix * trainFeats, iftMatrix * trainLabels);
double iftLogRegLoss(iftLogReg * lg, iftMatrix * featMatrix, iftMatrix * labelMatrix);
iftLogReg * iftLogRegTrainBySGD(iftDataSet * Z); 
void iftLogRegSGDClassifyCalcRow(iftDataSet * Z, int sample, iftMatrix * featMatrix, int row, iftMatrix * pMatrix);
void iftLogRegSGDClassify(iftLogReg * lg, iftDataSet * Z);
void iftLogRegSGDClassifySample(iftLogReg * lg, iftDataSet * Z, int sample);
// external private functions
iftLogReg* iftCreateLogReg();
float iftFindMinimumCostAndClassifySample(  iftCplGraph *graph, iftDataSet *Ztest, int t); // iftClassification.c

// ----- Public Functions -----
iftActiveClassifier * iftCreateActiveClassifier(
    iftClassifierAlgorithm algo,
    iftDataSet * Z,
    iftUnlabeledSelector * semisupSelector)
{
  iftActiveClassifier * classifier = calloc(1, sizeof(*classifier));
  classifier->algo = algo;
  classifier->isTrained = false;
  classifier->OPFClassifier = NULL;
  classifier->LogRegClassifier = NULL;
  classifier->Z = Z;
  classifier->Zlabeled = NULL;
  classifier->Zunlabeled = NULL;
  classifier->semisupSelector = semisupSelector;
  return classifier;
}

void iftTrainActiveClassifier(iftActiveClassifier * classifier)
{
  // Select labeled and unlabeled samples
  iftResetActiveClassifierDataSets(classifier);
  classifier->Zlabeled = iftExtractSamples(classifier->Z, IFT_TRAIN);
  if (iftIsSemiSupClassifier(classifier))
    classifier->Zunlabeled = iftExtractUnlabeledSemisupSamples(classifier);

  // Train classifier
  iftResetActiveClassifierClassifiers(classifier);
  switch (classifier->algo) {
    case OPFSUP_CLASSIFIER:
      iftTrainActiveClassifierOPFSup(classifier);
      break;
    case OPFSEMI_CLASSIFIER:
      iftTrainActiveClassifierOPFSemi(classifier);
      break;
    case LOGREG_CLASSIFIER:
      iftTrainActiveClassifierLogReg(classifier);
      break; 
    default:
      break;
  };
  classifier->isTrained = true;
}

void iftActiveClassifierClassifySample(iftActiveClassifier * classifier, iftDataSet * Ztest, int sample)
{
  switch (classifier->algo) {
    case OPFSUP_CLASSIFIER:
    case OPFSEMI_CLASSIFIER:
      iftFindMinimumCostAndClassifySample(classifier->OPFClassifier, Ztest, sample);
      break;
    case LOGREG_CLASSIFIER:
      iftLogRegSGDClassifySample(classifier->LogRegClassifier, Ztest, sample);
      break;
    default:
      break;
  };
}

void iftActiveClassifierClassifyDataSet(iftActiveClassifier * classifier, iftDataSet * Ztest)
{
  switch (classifier->algo) {
    case OPFSUP_CLASSIFIER:
    case OPFSEMI_CLASSIFIER:
      iftClassify(classifier->OPFClassifier, Ztest);
      break;
    case LOGREG_CLASSIFIER:
      iftLogRegSGDClassify(classifier->LogRegClassifier, Ztest);
    default:
      break;
  };
}

void iftDestroyActiveClassifier(iftActiveClassifier ** classifier)
{
  if (classifier == NULL)
   return;
  
  iftActiveClassifier * aux = *classifier; 
  if (aux != NULL) {
    iftResetActiveClassifierDataSets(aux);
    iftResetActiveClassifierClassifiers(aux);
    iftDestroyUnlabeledSelector(&(aux->semisupSelector));
    free(aux);
    *classifier = NULL;
  }
}

// ----- Private Functions -----
void iftTrainActiveClassifierOPFSup(iftActiveClassifier * classifier)
{
  iftDataSet * Ztrain = iftCopyDataSet(classifier->Zlabeled, true);
  classifier->OPFClassifier = iftCreateCplGraph(Ztrain);
  iftSupTrain(classifier->OPFClassifier);
}

void iftTrainActiveClassifierOPFSemi(iftActiveClassifier * classifier)
{
  // Avoid merge dataset problems
  classifier->Zlabeled->ngroups = 0;
  classifier->Zunlabeled->ngroups = 0;
  iftError("iftSemiSupTrain has been updated and this code is outdated", "iftTrainActiveClassifierOPFSemi");
  //classifier->OPFClassifier = iftSemiSupTrain(classifier->Zlabeled, classifier->Zunlabeled, FALSE);
}

void iftTrainActiveClassifierLogReg(iftActiveClassifier * classifier) 
{
  classifier->LogRegClassifier = iftLogRegTrainBySGD(classifier->Zlabeled);
}

iftDataSet * iftExtractUnlabeledSemisupSamples(iftActiveClassifier * classifier)
{
  iftDataSet * Z = classifier->Z;
  iftDataSet * Zunlabeled = NULL;
  iftUnlabeledSelector * semisupSelector = classifier->semisupSelector;
  int nSelect = classifier->Zlabeled->nsamples * 2;

  iftList * unlabeledSelected =
    iftPickFromUnlabeledSelector(semisupSelector, nSelect, classifier);

  if (unlabeledSelected->n == 0) {
    // All samples are already labeled, convert to Supervised algorithm
    classifier->algo = OPFSUP_CLASSIFIER;
  } else {
    // Clean status label for extraction
    for (int s = 0; s < Z->nsamples; ++s)
      if (Z->sample[s].status != IFT_TRAIN)
        Z->sample[s].status = IFT_UNKNOWN;

    // Mark selected samples as IFT_TEST
    iftNode * listIter = unlabeledSelected->head;
    while (listIter != NULL) {
      int sample = iftRemoveListNode(unlabeledSelected, &listIter, false); 
      Z->sample[sample].status = IFT_TEST;
    }

    Zunlabeled = iftExtractSamples(Z, IFT_TEST);
    iftSetStatus(Zunlabeled, IFT_TRAIN);
  }

  iftDestroyList(&unlabeledSelected);
  return Zunlabeled;
}

bool iftIsSemiSupClassifier(iftActiveClassifier * classifier)
{
  return (classifier->algo == OPFSEMI_CLASSIFIER);
}

void iftResetActiveClassifierDataSets(iftActiveClassifier * classifier)
{
  iftDestroyDataSet(&(classifier->Zlabeled));
  iftDestroyDataSet(&(classifier->Zunlabeled));
}

void iftResetActiveClassifierClassifiers(iftActiveClassifier * classifier)
{
  if (classifier->OPFClassifier != NULL) {
    // For convenience, the classifier graph always holds its own copy of Z
    iftDestroyDataSet(&(classifier->OPFClassifier->Z)); 
    iftDestroyCplGraph(&(classifier->OPFClassifier));
  }
  // TODO add this NULL test inside destroy function
  if (classifier->LogRegClassifier != NULL)
    iftDestroyLogReg(&(classifier->LogRegClassifier));
}

/*   iftLogRegGetConditionalProb
 * featMatrix nsamples x nfeats+1
 * weight matrix nfeats+1 x nclasses
 * returns m = [ P(Y=0|x_0,theta) P(Y=1|x_0,theta) ...
 *              [ P(Y=1|x_1,theta) ...
 *              [ ...
 */
iftMatrix * iftLogRegGetConditionalProb(iftMatrix * featMatrix, iftMatrix * weightMatrix)
{
  /*  m = XW = [ theta_0*x_0 theta_1*x_0 ...
   *           [ theta_0*x_1 ...
   *           [ ...
   */
  iftMatrix * m = iftMultMatrices(featMatrix, weightMatrix);

  /* m_ij = e^(m_ij)
   * m = [ e^(theta_0*x_0) e^(theta_1*x_0) ...
   *     [ e^(theta_0*x_1) ...
   *     [ ...
   */
  for (int i = 0; i < m->n; ++i)
    m->val[i] = exp(m->val[i]);

  /* normalizationFactor[n=0 to nsamples-1] = sum_{k=0 to nclasses-1} (e^(theta_k * x_n)) */
  double * normalizationFactor = iftAllocDoubleArray(m->nrows);
  for (int i = 0; i < m->nrows; ++i) {
    normalizationFactor[i] = 0.0;
    for (int j = 0; j < m->ncols; ++j)
      normalizationFactor[i] += m->val[iftGetMatrixIndex(m, j, i)];
  }

  /* m = [ P(Y=0|x_0,theta) P(Y=1|x_0,theta) ...
   *     [ P(Y=0|x_1,theta) ...
   *     [ ...
   */
  for (int i = 0; i < m->nrows; ++i)
    for (int j = 0; j < m->ncols; ++j)
      m->val[iftGetMatrixIndex(m, j, i)] /= normalizationFactor[i];

  // Clean up
  free(normalizationFactor);

  return m;
}

iftMatrix * iftLogRegGrad(iftLogReg * lg, iftMatrix * featMatrix, iftMatrix * labelMatrix)
{
  // -- Calculate gradient function
  iftMatrix * gradMatrix = iftCreateMatrix(lg->nclasses, lg->nfeats);
  iftMatrix * pMatrix = iftLogRegGetConditionalProb(featMatrix, lg->coef);

  // pMatrix_ij = (P(Y=j|x_i,theta) - t_ij)
  for (int i = 0; i < pMatrix->nrows; ++i) { // sample 
    for (int j = 0; j < pMatrix->ncols; ++j) { // class
      int index = iftGetMatrixIndex(pMatrix, j, i);
      pMatrix->val[index] -= labelMatrix->val[index];
    }
  }

  // grad_{theta_ij} = sum{s=0 to nsamples-1} x_si * (P(Y=j|x_s) - t_sj)
  for (int i = 0; i < gradMatrix->nrows; ++i) { // feat
    for (int j = 0; j < gradMatrix->ncols; ++j) { // class
      double res = 0.0;
      for (int s = 0; s < featMatrix->nrows; ++s) { // sample
        double tmp = featMatrix->val[iftGetMatrixIndex(featMatrix, i, s)]; // x_si
        tmp *= pMatrix->val[iftGetMatrixIndex(pMatrix, j, s)]; // * (y_sj - t_sj)
        res += tmp;
      }
      gradMatrix->val[iftGetMatrixIndex(gradMatrix, j, i)] = res;
    }
  }

  iftDestroyMatrix(&pMatrix);
  return gradMatrix;
}

iftList * iftSelectMiniBatchSamples(iftRandomSelector * rs, int batchSize)
{
  iftList * samples = iftCreateList();
  while(batchSize-- && rs->currentSize) {
    int s = iftPickFromRandomSelector(rs, false);
    iftInsertListIntoTail(samples, s);
  }

  return samples;
}

void iftFillSampleBatch(iftList * selectedSamples, iftMatrix * batchFeats,
    iftMatrix * batchLabels, iftMatrix * trainFeats, iftMatrix * trainLabels)
{
  iftNode * listIter = selectedSamples->head;
  int batchIdx = 0;
  while (listIter) {
    int s = iftRemoveListNode(selectedSamples, &listIter, false);

    // Insert sample feats and class into batch
    for (int feat = 0; feat < batchFeats->ncols; ++feat) {
      batchFeats->val[iftGetMatrixIndex(batchFeats, feat, batchIdx)] =
        trainFeats->val[iftGetMatrixIndex(trainFeats, feat, s)];
    }
    for (int c = 0; c < batchLabels->ncols; ++c) {
      batchLabels->val[iftGetMatrixIndex(batchLabels, c, batchIdx)] =
        trainLabels->val[iftGetMatrixIndex(trainLabels, c, s)];
    }

    batchIdx++;
  }
}

double iftLogRegLoss(iftLogReg * lg, iftMatrix * featMatrix, iftMatrix * labelMatrix)
{
  double error = 0.0;
  iftMatrix * pMatrix = iftLogRegGetConditionalProb(featMatrix, lg->coef);
  for (int i = 0; i < pMatrix->nrows; ++i) { // sample
    for (int j = 0; j < pMatrix->ncols; ++j) { // class
      int idx = iftGetMatrixIndex(pMatrix, j, i);
      if (labelMatrix->val[idx] > IFT_EPSILON) // != 0
        error -= log((double)(pMatrix->val[idx]));
    }
  }

  iftDestroyMatrix(&pMatrix);
  return error;
}

iftLogReg * iftLogRegTrainBySGD(iftDataSet * Z)
{

  iftLogReg * lg = iftCreateLogReg();
  lg->nclasses = Z->nclasses;
  lg->nfeats = Z->nfeats + 1;
  lg->coef = iftCreateMatrix(lg->nclasses, lg->nfeats);

  // Initialize weight matrix
  for (int i = 0; i < lg->coef->n; ++i)
    lg->coef->val[i] = 0.0;

  iftMatrix* trainFeats = iftDataSetToFeatureMatrixHomogCoord(Z, IFT_TRAIN);
  iftMatrix* trainLabels = iftDataSetToLabelsMatrix(Z, IFT_TRAIN);

  // SGD loop initialization
    int nEpochs = 100;
    int batchSize = 50;
  double learningRate = 0.1;
  double momentumParam = 0.9;
  double previousLoss = DBL_MAX;
  // Momentum helper matrix
  iftMatrix * previousDelta = iftCreateMatrix(lg->coef->ncols, lg->coef->nrows);
  for (int i = 0; i < previousDelta->n; ++i)
    previousDelta->val[i] = 0.0;
  // Selector for random minibatch samples
  iftRandomSelector * rs = iftCreateRandomSelector(trainFeats->nrows);
  for (int i = 0; i < rs->totalSize; ++i)
    rs->nums[i] = i;

  for (int epoch = 0; epoch < nEpochs; ++epoch) {
    iftResetRandomSelector(rs);
    while (rs->currentSize) {
      // Get random batch
      iftList * randomSamples = iftSelectMiniBatchSamples(rs, batchSize);
      iftMatrix * batchFeats = iftCreateMatrix(lg->nfeats, randomSamples->n);
      iftMatrix * batchLabels = iftCreateMatrix(lg->nclasses, randomSamples->n);
      iftFillSampleBatch(randomSamples, batchFeats, batchLabels, trainFeats, trainLabels);

      // Calculate gradient of loss function
      iftMatrix * gradMatrix = iftLogRegGrad(lg, batchFeats, batchLabels);

      // Update weights given grad
      for (int i = 0; i < lg->coef->nrows; ++i) { // feat
        for (int j = 0; j < lg->coef->ncols; ++j) { // class
          int idx = iftGetMatrixIndex(lg->coef, j, i);
          double delta = momentumParam * previousDelta->val[idx]
            + learningRate * (double) gradMatrix->val[idx];
          lg->coef->val[idx] -= delta;
          previousDelta->val[idx] = delta;
        }
      }

      // Clean up
      iftDestroyList(&randomSamples);
      iftDestroyMatrix(&batchFeats);
      iftDestroyMatrix(&batchLabels);
      iftDestroyMatrix(&gradMatrix);
    }

    // Dynamically update learning rate based on improvement
    double loss = iftLogRegLoss(lg, trainFeats, trainLabels);
    if (loss/previousLoss >= 0.9995) {
        learningRate /= 10;
    }
    previousLoss = loss;

    // Early stop if too small
    if (learningRate < 0.0001)
      break;
  }

  iftDestroyMatrix(&trainFeats);
  iftDestroyMatrix(&trainLabels);
  iftDestroyMatrix(&previousDelta);
  iftDestroyRandomSelector(&rs);

  return lg;
}

void iftLogRegSGDClassifyCalcRow(iftDataSet * Z, int sample, iftMatrix * featMatrix, int row, iftMatrix * pMatrix)
{
  float weight = -FLT_MAX; 
  for (int c = 0; c < pMatrix->ncols; ++c) {
    float p = pMatrix->val[iftGetMatrixIndex(pMatrix, c, row)];
    if (p > weight) {
      weight = p;
      Z->sample[sample].label = c+1;
    }
  }
}

void iftLogRegSGDClassify(iftLogReg * lg, iftDataSet * Z)
{
  iftMatrix * featMatrix = iftDataSetToFeatureMatrixHomogCoord(Z, IFT_TEST);
  iftMatrix * pMatrix = iftLogRegGetConditionalProb(featMatrix, lg->coef);
  int i = 0;
  for (int s = 0; s < Z->nsamples; ++s) {
    if (Z->sample[s].status != IFT_TEST)
      continue;

    iftLogRegSGDClassifyCalcRow(Z, s, featMatrix, i, pMatrix);

    i++;
  }

  iftDestroyMatrix(&featMatrix);
  iftDestroyMatrix(&pMatrix);
}

void iftLogRegSGDClassifySample(iftLogReg * lg, iftDataSet * Z, int sample)
{
  // Build single row matrix for convenience
  iftMatrix * sampleHomogFeats = iftCreateMatrix(Z->nfeats+1,1);
  for (int i = 0; i < Z->nfeats; ++i)
    sampleHomogFeats->val[i] = Z->sample[sample].feat[i];
  sampleHomogFeats->val[Z->nfeats] = 1.0;

  iftMatrix * pMatrix = iftLogRegGetConditionalProb(sampleHomogFeats, lg->coef);
  iftLogRegSGDClassifyCalcRow(Z, sample, sampleHomogFeats, 0, pMatrix);

  iftDestroyMatrix(&sampleHomogFeats);
  iftDestroyMatrix(&pMatrix);
}

