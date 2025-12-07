#include <ift.h>
#include "Model.h"
#include "Tensor.h"
#include "iftHybridNetwork.h"
#include "iftHybridNetworkTF.h"

int main(int argc, char* argv[]) {
  if (argc < 7) {
    printf("Usage %s <svm.zip> <minibatch_size> <saved_model.pb> <error_data.npy> <test_dataset.zip> <perc_network = [0,1]>\n", argv[0]);
    return -1;
  }
  
  iftSVM *svm = iftReadSVM(argv[1]);
  int miniBatchSize = atol(argv[2]);
  Model model(argv[3]);
  iftMatrix *errorData = iftReadMatrix(argv[4]);
  iftDataSet *Z = iftReadDataSet(argv[5]);
  iftMatrix *pMatrix = iftGetHybridBinProbabilities(errorData, atof(argv[6]));
  //iftPrintMatrix(pMatrix);
  //iftPrintMatrix(errorData);

  // Network info
  std::vector<float> data(PHN::imgLen * miniBatchSize);
  Tensor net_input{model, "x"};
  Tensor net_output{model, "Identity"};

  iftFileSet *fileset = (iftFileSet *) Z->ref_data;
  iftSetStatus(Z, IFT_TEST);

  // Fill batch and classify loop
  int nSamples = fileset->n;
  int nClasses = net_output.get_shape()[1];
  int batchNum = 1;

  int netCount = 0;
  std::vector<int> batchSampleIndex(miniBatchSize);
  std::vector<int> batchSampleSVMLabel(miniBatchSize);
  std::vector<int> batchSampleNetLabel(miniBatchSize);
  std::vector<int> batchSampleTrueLabel(miniBatchSize);
  int batchCounter = 0;

  for (int sample = 0; sample < Z->nsamples; ++sample) {
	  int svmLabel;
	  float svmProb;

	  iftSVMClassifyOVO_ProbabilitySingleSample(svm, Z->sample[sample].feat,
			  Z->nfeats, &svmLabel, &svmProb);
	  Z->sample[sample].label = svmLabel;

//	  printf("Sample %d - TrueLabel=%d, SVMLabel=%d\n", sample, Z->sample[sample].truelabel, svmLabel);

	  if (iftHybridDecision(svmLabel, svmProb, pMatrix)) {
		  netCount += 1;
		  const char *imgPath = fileset->files[sample]->path;
		  iftVggLoadImgFromPath_TF(data, batchCounter, imgPath);

		  batchSampleIndex[batchCounter] = sample;
		  batchSampleTrueLabel[batchCounter] = Z->sample[sample].truelabel;
		  batchSampleSVMLabel[batchCounter] = svmLabel;

		  batchCounter++;

		  if (batchCounter == miniBatchSize || sample == (Z->nsamples - 1)) {
			  // fill input with batch
			  net_input.set_data(data);

			  printf("Starting forward for batch %d (last sample index %d)\n", batchNum++, sample);
			  timer *batchT1 = iftTic();
			  model.run(net_input, net_output);
			  timer *batchT2 = iftToc();
			  float batchTime = iftCompTime(batchT1, batchT2);
			  printf("Batch forward finished in %fms (%fms per sample)\n", batchTime, batchTime / batchCounter);

			  // Fill in predictions
			  for (int i = 0; i < batchCounter; ++i) {
				  int netLabel = 0;
				  float netMaxVal = FLT_MIN;
				  for (int label = 0; label < nClasses; ++label) {
					  float outputVal = net_output.get_data<float>()[i * nClasses + label];
					  if (outputVal > netMaxVal) {
						  netLabel = label;
						  netMaxVal = outputVal;
					  }
				  }
				  netLabel++;
				  batchSampleNetLabel[i] = netLabel;
				  Z->sample[batchSampleIndex[i]].label = netLabel;

//				  printf("Sample batch %d - Index=%d, TrueLabel=%d, SVMLabel=%d, NetLabel=%d\n",
//						  i, batchSampleIndex[i], batchSampleTrueLabel[i], batchSampleSVMLabel[i], batchSampleNetLabel[i]);
			  }

			  batchCounter = 0;
		  }
	  }
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
  iftDestroyMatrix(&errorData);
  iftDestroyDataSet(&Z);
  iftDestroyMatrix(&pMatrix);
  free(nErrorsPerClass);
  free(nSamplesPerClass);

  return 0;
}
