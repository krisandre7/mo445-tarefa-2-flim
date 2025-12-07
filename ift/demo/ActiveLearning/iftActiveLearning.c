#include "ift.h"
#include <assert.h>

/**
 * Active learning demo that uses clustering, boundary reduction, MST and boundary samples.
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
	fread(&version, 1, 1, fp);
	if (version != (int8_t) 0xff) {
	  rewind(fp);
	}

	/* Read # of samples, classes and feats*/

	fread(&nsamples, sizeof(int32_t), 1, fp);
	fread(&nclasses, sizeof(int32_t), 1, fp);
	fread(&nfeats, sizeof(int32_t), 1, fp);

	Z = iftCreateDataSet(nsamples, nfeats);
	Z->nclasses = nclasses;
	Z->ntrainsamples = nsamples;

	/* Read features of each sample */

	char prefix[256];
	for (s=0; s < nsamples; s++){

	  fread(&distind, sizeof(int32_t), 1, fp); /* Pre-computed distances are not being used this way. */
	  fread(&(Z->sample[s].truelabel), sizeof(int32_t), 1, fp);

	  if (version == (int8_t) 0xff) {
		  u_int32_t textSize;
		  fread(&textSize, sizeof(u_int32_t), 1, fp);
		  if (textSize > 0) {
			  fread(prefix, sizeof(u_int8_t), textSize, fp);
		  }
	  }

	  fread(Z->sample[s].feat, sizeof(float), nfeats, fp);

	  Z->sample[s].id = s; /* for distance table access */
	  Z->sample[s].status = IFT_TRAIN;
	}
	fclose(fp);

	return(Z);
}

int main(int argc, char *argv[])
{
  iftDataSet      *Z=NULL;
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
    iftError("Usage: iftActiveLearning <OPF dataset.dat> <kmax_perc [0,1]>","main");

#ifdef USE_MYOPFDATASET
  Z = readMyOPFDataSet(argv[1]);
#else
  Z = iftReadOPFDataSet(argv[1]);
#endif
  printf("Read dataset with %d samples.\n", Z->nsamples);

  srand(time(NULL));

  t1 = iftTic();

  graph = iftUnsupLearn(Z, atof(argv[2]), iftNormalizedCut, 10);
  iftUnsupClassify(graph, Z);

  t2 = iftToc();

  fprintf(stdout,"clustering in %f ms with %d groups\n",iftCompTime(t1,t2), Z->nlabels);
  printf("%d nodes in the clustered graph (k=%d kmax=%d).\n", graph->nnodes, graph->k, graph->kmax);

  int i;
  for (i = 0; i < Z->nsamples; i++) {
	  Z->sample[i].status = NIL;
  }
  Z->ntrainsamples = 0;

  /* check if the graph has roots from all classes */
  int cls[Z->nclasses + 1];
  memset(cls, 0, sizeof(int) * Z->nclasses);
  int nroots = 0;
  for (i = 0; i < graph->nnodes; i++) {
	if (i == graph->node[i].root) { // if node is root
	  int sampleIdx = graph->node[i].sample;
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

  iftDataSet *reducedZ = iftGraphBoundaryReduction(graph, Z);
  printf("%d samples in the reduced set.\n", reducedZ->nsamples);

  iftMST *reducedMST = iftCreateMST(reducedZ);

  for (i = 0; i < reducedZ->nsamples; i++) {
	  reducedZ->sample[i].status = NIL;
	  reducedZ->sample[i].truelabel = NIL;
  }
  reducedZ->ntrainsamples = 0;

  iftSortNodesByWeightMST(reducedMST, DECREASING);

  iftCplGraph *trainGraph = iftCreateCplGraph(Z);
  iftSupTrain(trainGraph);
  printf("Trained graph with %d root nodes.\n", trainGraph->nnodes);

  int iteration = 1;
  int nSamplesShown = 0;
  int nSamplesCorrected = 0;
  do {
	  printf("Iteration %d\n", iteration++);
	  iftClassify(trainGraph, reducedZ);

	  iftSet *selectedSamples = iftSelectSamplesForUserLabelingMST(reducedMST, Z->nclasses * 2); // select 2 * number of classes
	  int userCorrections = 0;
	  int sampleIdx;
	  int nsamples = 0;
	  while ((sampleIdx = iftRemoveSet(&selectedSamples)) != NIL) {
		  nsamples++;
		  if (Z->sample[reducedZ->sample[sampleIdx].id].truelabel != reducedZ->sample[sampleIdx].label) { // if sample was misclassified
			  userCorrections++;
			  reducedZ->sample[sampleIdx].truelabel = Z->sample[reducedZ->sample[sampleIdx].id].truelabel;
			  reducedZ->sample[sampleIdx].label = reducedZ->sample[sampleIdx].truelabel;
		  } else {
			  reducedZ->sample[sampleIdx].truelabel = reducedZ->sample[sampleIdx].label;
		  }
		  reducedZ->sample[sampleIdx].status = IFT_TRAIN;
		  reducedZ->ntrainsamples++;
		  Z->sample[reducedZ->sample[sampleIdx].id].status = IFT_TRAIN;
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

	  iftClassify(trainGraph, Z);
	  float accuracy = iftClassifAccuracy(Z);
	  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy * 100.0f);
	  printf("\n");
  } while (1);

  printf("\n");
  printf("%d samples shown to the user.\n", nSamplesShown);
  printf("%d samples corrected by the user.\n", nSamplesCorrected);

  iftWriteOPFDataSet(trainGraph->Z, "training_active_learning.opf");
  printf("\n");
  printf("Saving OPF dataset training_active_learning.opf\n");

  iftDestroyCplGraph(&trainGraph);

  iftDestroyMST(&reducedMST);
  iftDestroyDataSet(&reducedZ);
  iftDestroyDataSet(&Z);
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
