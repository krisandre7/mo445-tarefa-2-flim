#include <assert.h>
#include <stdbool.h>
#include "ift.h"

/**
 * Active learning demo of the Root Distance-Based Sampling method.
 **/

//#define USE_MYOPFDATASET

iftDataSet *readMyOPFDataSet(char *filename) {
	iftDataSet *Z  = NULL;
	FILE       *fp = NULL;
	int  nsamples, nclasses, nfeats, distind, s;

	fp = fopen(filename,"rb");
	if (fp == NULL)
		iftError(MSG2,"readMyOPFDataSet");

	char version;
	size_t ret = fread(&version, 1, 1, fp);
	if (ret != 1) iftError("Error reading version.", "readMyOPFDataSet");
	if (version != (int8_t) 0xff) {
	  rewind(fp);
	}

	/* Read # of samples, classes and feats*/

	ret = fread(&nsamples, sizeof(int32_t), 1, fp);
	if (ret != 1) iftError("Error reading nsamples.", "readMyOPFDataSet");
	ret = fread(&nclasses, sizeof(int32_t), 1, fp);
	if (ret != 1) iftError("Error reading nclasses.", "readMyOPFDataSet");
	ret = fread(&nfeats, sizeof(int32_t), 1, fp);
	if (ret != 1) iftError("Error reading nfeats.", "readMyOPFDataSet");

	Z = iftCreateDataSet(nsamples, nfeats);
	Z->nclasses = nclasses;

	/* Read features of each sample */

	char prefix[256];
	for (s=0; s < nsamples; s++){

		ret = fread(&distind, sizeof(int32_t), 1, fp);
		if (ret != 1) iftError("Error reading distind.", "readMyOPFDataSet");
		ret = fread(&(Z->sample[s].truelabel), sizeof(int32_t), 1, fp);
		if (ret != 1) iftError("Error reading sample class.", "readMyOPFDataSet");

	  if (version == (int8_t) 0xff) {
		  u_int32_t textSize;
		  ret = fread(&textSize, sizeof(u_int32_t), 1, fp);
			if (ret != 1) iftError("Error reading textSize.", "readMyOPFDataSet");
		  if (textSize > 0) {
		  		ret = fread(prefix, sizeof(u_int8_t), textSize, fp);
		  		if (ret != textSize) iftError("Error reading prefix.", "readMyOPFDataSet");
		  }
	  }

	  ret = fread(Z->sample[s].feat, sizeof(float), nfeats, fp);
		if (ret != nfeats) iftError("Error reading nfeats.", "readMyOPFDataSet");

	  Z->sample[s].id = s; /* for distance table access */
	  Z->sample[s].status = IFT_TRAIN;

	}
	fclose(fp);

	Z->ntrainsamples = Z->nsamples;

	return(Z);
}

int main(int argc, char *argv[])
{
  iftDataSet      *Z=NULL;
  iftDataSet      *dataset=NULL;
  iftDataSet      *Test=NULL;
  iftKnnGraph     *graph=NULL;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

#ifndef _OPENMP
  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif

  /*--------------------------------------------------------*/

  if (argc != 3)
    iftError("Usage: iftActiveLearningRDS <OPF dataset.dat> <kmax_perc [0,1]>","main");

#ifdef USE_MYOPFDATASET
  dataset = readMyOPFDataSet(argv[1]);
#else
  dataset = iftReadOPFDataSet(argv[1]);
#endif
  printf("Read dataset with %d samples.\n", dataset->nsamples);

  srand(time(NULL));

  iftSelectSupTrainSamples(dataset, 0.80);
  Z = iftExtractSamples(dataset,IFT_TRAIN);
  Test = iftExtractSamples(dataset,IFT_TEST);


  t1 = iftTic();
  iftSelectUnsupTrainSamples(Z,1.0);
  graph = iftUnsupLearn(Z, atof(argv[2]), iftNormalizedCut, 10);
  iftUnsupClassify(graph, Z);

  t2 = iftToc();

  fprintf(stdout,"clustering in %f ms with %d groups\n",iftCompTime(t1,t2), Z->nlabels);
  printf("%d nodes in the clustered graph (k=%d kmax=%d).\n", graph->nnodes, graph->k, graph->kmax);

  int i;
  for (i = 0; i < Z->nsamples; i++) {
	  Z->sample[i].status = IFT_TEST;
  }
  Z->ntrainsamples = 0;

  /* check if the graph has roots from all classes */
  int cls[Z->nclasses + 1];
  memset(cls, 0, sizeof(int) * Z->nclasses);
  int nroots = 0;
  for (i = 0; i < graph->nnodes; i++) {
  	int sampleIdx = graph->node[i].sample;
  	if (i == graph->node[i].root) { // if node is root
			cls[Z->sample[sampleIdx].truelabel] = 1;
			Z->sample[sampleIdx].status = IFT_TRAIN; // select root for training
			nroots++;
		}
  }
  assert(nroots == Z->nlabels);
  for (i = 1; i <= Z->nclasses; i++) {
		if (cls[i] == 0) {
			fprintf(stderr, "Missing cluster with root class %d\n", i);
			iftDestroyDataSet(&Z);
			iftDestroyKnnGraph(&graph);
			exit(-1);
		}
  }
  Z->ntrainsamples = Z->nlabels;

  int *nNodes;
  int **nodeLists = iftGetNodesPerClusterOrderedByDistanceToRoot(graph, &nNodes);

  // create train graph with root nodes (status = IFT_TRAIN)
  iftCplGraph *trainGraph = iftCreateCplGraph(Z);
  iftSupTrain(trainGraph);
  printf("Trained graph with %d root nodes.\n", trainGraph->nnodes);

  int iteration = 1;
  int nSamplesShown = 0;
  int nSamplesCorrected = 0;
  do {
 		printf("Iteration %d\n", iteration++);

 		// select at most (2 * number of classes) samples without IFT_TRAIN status
 		iftSet *selectedSamples = iftGetRootDistanceBasedSamples(
 				trainGraph, Z, nodeLists, nNodes, IFT_TRAIN, 1, Z->nclasses * 2);

 		int userCorrections = 0;
 		int sampleIdx;
 		int nsamples = 0;
 		while ((sampleIdx = iftRemoveSet(&selectedSamples)) != NIL) {
 			// set sample status to IFT_TRAIN
 			Z->sample[sampleIdx].status = IFT_TRAIN;
			nsamples++;
 			if (Z->sample[sampleIdx].truelabel != Z->sample[sampleIdx].label) { // if sample was misclassified
				userCorrections++;
				Z->sample[sampleIdx].label = Z->sample[sampleIdx].truelabel;
 			}
 			Z->ntrainsamples++;
 		}
 		printf("%d samples selected for user evaluation.\n", nsamples);
 		printf("%d user corrections.\n", userCorrections);

 		if (nsamples == 0) break;

 		iftDestroyCplGraph(&trainGraph);

 		nSamplesCorrected += userCorrections;
 		nSamplesShown += nsamples;

 		trainGraph = iftCreateCplGraph(Z);
 		iftSupTrain(trainGraph);
 		printf("Trained graph with %d nodes.\n", trainGraph->nnodes);

 		//iftClassify(trainGraph, Z);
 		iftClassify(trainGraph, Test);
 		//float accuracy = iftClassifAccuracy(Z);
 		float accuracy = iftTruePositives(Test);
 		printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy * 100.0f);
 		printf("\n");
  } while (1);

  free(nNodes);
  for (i = 1; i <= Z->nlabels; i++) {
 		free(nodeLists[i]);
  }
  free(nodeLists);

  printf("\n");
  printf("%d samples shown to the user.\n", nSamplesShown);
  printf("%d samples corrected by the user.\n", nSamplesCorrected);

  iftWriteOPFDataSet(trainGraph->Z, "training_active_learning.opf");
  printf("\n");
  printf("Saving OPF dataset training_active_learning.opf\n");

  iftDestroyCplGraph(&trainGraph);

  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&Test);
  iftDestroyDataSet(&dataset);
  iftDestroyKnnGraph(&graph);

  /* ---------------------------------------------------------- */

#ifndef _OPENMP
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);
#endif

  return(0);
}
