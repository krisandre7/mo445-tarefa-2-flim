#include <ift.h>

void iftSVMClassifyOVO_ProbabilitySingleSample(  iftSVM *svm, float *feats, int nFeats, int *outLabel, float *outProb)
{
  assert(nFeats == svm->model[0]->l); // TODO check if that's the correct parameter 
  int nClasses = svm->model[0]->nr_class;
  int precomp = svm->params->kernel_type == PRECOMPUTED;
  double *prob_estimates = (double *) malloc(nClasses*sizeof(double));

  if (svm->nmodels == 0)
    iftError("SVM not trained yet", "iftSVMClassifyOVO_ProbabilitySingleSample");

  if (svm->truelabel != NULL)
    iftError("This is not a p-SVM", "iftSVMClassifyOVO_ProbabilitySingleSample");

  // temporary buffer to get predictions from libsvm
  svmNode *x = (svmNode *) iftAlloc(precomp + nFeats + 1, sizeof(svmNode));

  iftSampleToSvmSample(feats, x, 0, nFeats, precomp);
  *outLabel = (int) svm_predict_probability(svm->model[0], x, prob_estimates);
  *outProb = prob_estimates[(*outLabel) - 1];

  free(probEstimates);
  iftFree(x);
}

int main(int argc, char* argv[])
{
  if (argc < 5) {
    printf("Usage %s <trained svm> <cnn arch> <trained cnn weights> <error rates>\n", argv[0]);
    return -1;
  

  iftSVM *svm = iftReadSVM(argv[1]);
  iftNeuralNetwork *net = iftCreateWeightedNeuralNetworkFromJson(argv[2], argv[3]); 
  iftMatrix *pMatrix = iftReadRawMatrix(argv[4]);

  // TODO SVM classify single sample
  // TODO Use error rate information to select if network is used
  

  return 0;
}
