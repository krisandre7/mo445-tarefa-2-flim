#include "ift.h"
#include <assert.h>

/**
 * Active learning demo that uses clustering, boundary reduction, MST and boundary samples, using SVM.
 **/

iftDataSet *iftExtractSamplesNotStatus(iftDataSet *Z, uchar status)
{
	int         i, s, t, nsamples;
	iftDataSet *Z1;

	nsamples=0;
	for (s=0; s < Z->nsamples; s++)
		if (Z->sample[s].status != status){
			nsamples++;
		}

	if (nsamples == 0)
		iftError("There are no samples from the desired status","iftExtractSamples");

	Z1               = iftCreateDataSet(nsamples,Z->nfeats);
	Z1->nclasses     = Z->nclasses;
	Z1->iftArcWeight = Z->iftArcWeight;
	Z1->function_number = Z->function_number;
	Z1->ref_data     = Z->ref_data;

	t=0;
	for (s=0; s < Z->nsamples; s++)
		if (Z->sample[s].status != status){
			Z1->sample[t].id     = Z->sample[s].id;
			Z1->sample[t].truelabel  = Z->sample[s].truelabel;
			Z1->sample[t].status = status;
			for (i=0; i < Z->nfeats; i++)
				Z1->sample[t].feat[i] = Z->sample[s].feat[i];
			t++;
		}

	Z1->fsp = iftCopyFeatSpaceParam(Z);

	for (i=0; i < Z->nfeats; i++)
		Z1->alpha[i] = Z->alpha[i];

	return(Z1);
}

int main(int argc, char *argv[])
{
  iftDataSet      *dataset=NULL;
  timer           *t1=NULL,*t2=NULL;
  int i, j;

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

  if (argc != 4)
    iftError("Usage: iftActiveLearning <OPF dataset.dat> <kmax_perc [0,1]> <output [csv]>","main");

  dataset = iftReadOPFDataSet(argv[1]);

  printf("Read dataset with %d samples.\n", dataset->nsamples);
  FILE *f = fopen(argv[3], "w");

  srand(time(NULL));

  /* Variables for accurary and time */
  timer *tstart, *tfinish;
  timer *tic,*toc;
  float *time_selection, *time_train, *time_cls, *accuracy, *num_showed, *num_annotated, *time_iter, *ntrainsamples;
  float accuracy_iter;
  int max_iteration = 50;
  int nrun = 8;
  time_selection = iftAllocFloatArray(max_iteration);
  time_train = iftAllocFloatArray(max_iteration);
  time_cls = iftAllocFloatArray(max_iteration);
  accuracy = iftAllocFloatArray(max_iteration);
  num_showed = iftAllocFloatArray(max_iteration);
  num_annotated = iftAllocFloatArray(max_iteration);
  time_iter = iftAllocFloatArray(max_iteration);
  ntrainsamples = iftAllocFloatArray(max_iteration);

  for (j = 0; j < nrun; j++) {
	  iftKnnGraph     *graph=NULL;
	  iftDataSet      *Z=NULL;
      iftDataSet      *Test=NULL;
	  iftSelectSupTrainSamples(dataset, 0.80);
	  Z = iftExtractSamples(dataset,IFT_TRAIN);

	  Test = iftExtractSamples(dataset,IFT_TEST);

	  printf("nRun: %d\n", j);

	  int s;
	  int all_cls = 0;
	  float k = atof(argv[2]);

	  while(all_cls == 0) {
		  t1 = iftTic();
		  iftSelectUnsupTrainSamples(Z,1.0);
		  graph = iftUnsupLearn(Z, k, iftNormalizedCut, 10);
		  iftUnsupClassify(graph, Z);

		  t2 = iftToc();

		  fprintf(stdout,"clustering in %f ms with %d groups\n",iftCompTime(t1,t2), Z->nlabels);
		  printf("%d nodes in the clustered graph (k=%d kmax=%d).\n", graph->nnodes, graph->k, graph->kmax);


		  for (i = 0; i < Z->nsamples; i++) {
			Z->sample[i].status = NIL;
		  }
		  Z->ntrainsamples = 0;

		  /* check if the graph has roots from all classes */
		  int cls[Z->nclasses + 1];
		  memset(cls, 0, sizeof(int) * (Z->nclasses+1));
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
		  int cnt = 0;
		  for (i = 1; i <= Z->nclasses; i++) {
			if (cls[i] == 0) {
			  fprintf(stderr, "Missing cluster with root class %d\n", i);
			  k = k - 0.0005;
			  break;
			}else{
				cnt++;
			}
		  }
		  if(cnt == Z->nclasses)
			  all_cls = 1;
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


	  /* Train by SVM */

	  float cSVM = 1.0;
	  iftSVM      *svm   = NULL;
	  iftDataSet *Ztrain[2], *Ztest[2], *Ztmp;

	  Ztmp = iftCopyDataSet(Z, true);
	  printf("ntrainsamples %d\n",Ztmp->ntrainsamples);
	  Ztrain[0] = iftExtractSamples(Ztmp,IFT_TRAIN);
	  Ztest [0] = iftExtractSamplesNotStatus(Ztmp,IFT_TRAIN);
	  iftSetStatus(Ztest[0],IFT_TEST);
	  iftDestroyDataSet(&Ztmp);

	  Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
	  iftDestroyDataSet(&Ztrain[0]);
	  Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
	  iftDestroyDataSet(&Ztest [0]);

	  svm=iftCreateLinearSVC(cSVM);
	  iftSVMTrainOVA(svm, Ztrain[1]);

	  iftDestroySVM(svm);
	  iftDestroyDataSet(&Ztest[1]);

	  int iteration = 0;
	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;
	  do {
		  tstart = iftTic();
		  printf("Iteration %d\n", iteration+1);

		  tic = iftTic();
		  /* Classify reducedZ */
		  Ztest [0] = iftExtractSamplesNotStatus(reducedZ,IFT_TRAIN);
		  iftSetStatus(Ztest[0],IFT_TEST);
		  Ztest [1] = iftNormalizeTestDataSet(Ztrain[1], Ztest[0]);
		  iftDestroyDataSet(&Ztest [0]);
		  svm=iftCreateLinearSVC(cSVM);
		  iftSVMTrainOVA(svm, Ztrain[1]);
		  iftSVMLinearClassifyOVA(svm, Ztest[1], Ztrain[1], IFT_TEST);

		  /* Copy labels */
		  int ind = 0;
		  for (s=0; s < reducedZ->nsamples; s++) {
			  if(reducedZ->sample[s].status != IFT_TRAIN){
				  reducedZ->sample[s].label = Ztest[1]->sample[ind].label;
				  ind++;
			  }
		  }
		  iftDestroyDataSet(&Ztrain[1]);
		  iftDestroyDataSet(&Ztest[1]);
		  iftDestroySVM(svm);

		  /* Select samples for user labeling */
		  iftSet *selectedSamples = iftSelectSamplesForUserLabelingMST(reducedMST, Z->nclasses * 2); // select 2 * number of classes
		  toc = iftToc();
		  time_selection[iteration] += iftCompTime(tic,toc) / (float)nrun;

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
		  ntrainsamples[iteration] = (float)Z->ntrainsamples / (float)nrun;
		  printf("# train samples %d.\n", Z->ntrainsamples);
		  
		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
		  num_showed[iteration] += nsamples / (float)nrun;
		  num_annotated[iteration] += userCorrections / (float)nrun;

		  if (nsamples == 0) break;

		  nSamplesCorrected += userCorrections;
		  nSamplesShown += nsamples;

		  /* Train Z */

		  tic = iftTic();
		  Ztmp = iftCopyDataSet(Z, true);
		  Ztrain[0] = iftExtractSamples(Ztmp,IFT_TRAIN);
		  //Ztest [0] = iftExtractSamplesNotStatus(Ztmp,IFT_TRAIN);
		  Ztest [0] = iftCopyDataSet(Test, true);
		  iftSetStatus(Ztest[0],IFT_TEST);
		  iftDestroyDataSet(&Ztmp);

		  Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
		  iftDestroyDataSet(&Ztrain[0]);
		  Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
		  iftDestroyDataSet(&Ztest [0]);

		  svm=iftCreateLinearSVC(cSVM);
		  iftSVMTrainOVA(svm, Ztrain[1]);
		  toc = iftToc();
		  time_train[iteration] += iftCompTime(tic,toc) / (float)nrun;

		  /* Classify Z */
		  tic = iftTic();
		  iftSVMLinearClassifyOVA(svm, Ztest[1], Ztrain[1], IFT_TEST); // Classification
		  accuracy_iter = iftClassifAccuracy(Ztest[1]);
		  accuracy[iteration] += accuracy_iter  / (float)nrun;
		  toc = iftToc();
		  time_cls[iteration] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
		  tfinish = iftToc();
		  time_iter[iteration] += iftCompTime(tstart,tfinish) / (float)nrun;

		  iftDestroyDataSet(&Ztest[1]);
		  iftDestroySVM(svm);

		  iteration++;
	  } while (iteration < max_iteration);

	  iftDestroyKnnGraph(&graph);
	  iftDestroyDataSet(&reducedZ);
	  //iftDestroyMST(&reducedMST);
	  iftDestroyDataSet(&Test);
  	  //iftDestroyDataSet(&Z);

  }
  
  for (j = 0; j < max_iteration; j++) {
  	fprintf(f,"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", time_selection[j], time_train[j], time_cls[j], accuracy[j], num_showed[j], num_annotated[j], time_iter[j], ntrainsamples[j]);
  }

  fclose(f);

  printf("\n");
  printf("Finish exp. Active Learning SVM\n");

  iftDestroyDataSet(&dataset);
  printf("Finish free\n");

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
