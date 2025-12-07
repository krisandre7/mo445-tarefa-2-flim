#include "ift.h"
#include <assert.h>

/**
 * Active learning demo that uses clustering, and Maximum Uncertainty.
 **/

iftSet *  iftSelectSamplesForUserLabelingMU(iftCplGraph *graph, iftDataSet *Z, int n)
{
	iftSet *selectedNodes = NULL;
	int u, s, t, i, setSize, label = 0,label2 = 0;
	float tmp, weight, minCost, minCost2;
	iftDataSet *Z1=graph->Z;

	if (Z1->nfeats != Z->nfeats)
	  {
	    char msg[200];
	    sprintf(msg,"Incompatible datasets (%d/%d)",Z1->nfeats,Z->nfeats);
	    iftError(msg,"iftClassify");
	  }

#pragma omp parallel for schedule(dynamic) shared(Z,Z1,graph,iftDist) private(t,i,u,s,weight,minCost,label,tmp)
	for (t = 0; t < Z->nsamples; t++)
	{
		if ((Z==Z1)&&(Z->sample[t].status==IFT_TRAIN))
			continue;

		i       = 0;
		u       = graph->ordered_nodes[i];
		s       = graph->node[u].sample;
		if (iftDist == NULL)
			weight  = Z1->iftArcWeight(Z1->sample[s].feat,Z->sample[t].feat,Z1->alpha,Z1->nfeats);
		else
			weight  = iftDist->distance_table[Z1->sample[s].id][Z->sample[t].id];

		minCost = MAX(graph->pathval[u], weight);
		minCost2 = 9999.0;

		label   = Z1->sample[s].label;
		label2 = 0;


		while((i < graph->nnodes-1)&&
				(minCost > graph->pathval[graph->ordered_nodes[i+1]])){

			u  = graph->ordered_nodes[i+1];
			s  = graph->node[u].sample;

			if (iftDist == NULL)
				weight = Z1->iftArcWeight(Z1->sample[s].feat,Z->sample[t].feat,Z1->alpha,Z1->nfeats);
			else
				weight  = iftDist->distance_table[Z1->sample[s].id][Z->sample[t].id];

			tmp = MAX(graph->pathval[u], weight);
			if(tmp < minCost){
				minCost2 = minCost;
				label2 = label;

				minCost = tmp;
				label   = Z1->sample[s].label;
			}else if(tmp < minCost2){
				minCost2 = tmp;
				label2 = Z1->sample[s].label;
			}
			i++;
		}

		if(label2 != 0 && minCost2 > 0 && label != label2){
			float ratio = minCost / minCost2;
			if(ratio > 0.8){
				printf("t: %d, ratio: %f, minCost: %f %d, minCost2: %f %d\n", t, ratio, minCost, label, minCost2, label2);
				if(setSize < n) {
					setSize += iftUnionSetElem(&selectedNodes, t);
				}
			}
		}
	}
	return(selectedNodes);
}

int iftSelectSupTrainSamples1(iftDataSet *Z, float train_perc)
{
  int s, *sample=NULL, *count=NULL, i;
  int t, high, nsamples, c, *class_size=NULL;

  if (Z->nclasses == 0)
    iftError("There are no classes","iftSelectSupTrainSamples");

  if ((train_perc <= 0.0)||(train_perc > 1.0))
    iftError("Invalid percentage of training samples","iftSelectSupTrainSamples");

  // Reset status

  iftSetStatus(Z, IFT_TEST);

  // Verify if it is the trivial case of selecting all samples.

  if (train_perc == 1.0) {
    for (s=0; s < Z->nsamples; s++)
      Z->sample[s].status = IFT_TRAIN;
    Z->ntrainsamples = Z->nsamples;
    return(Z->nsamples);
  }

  // Compute the number of training samples

  Z->ntrainsamples = (int) (train_perc*Z->nsamples);
  if (Z->ntrainsamples == 0)
    iftError("Percentage is too low for this dataset","iftSelectSupTrainSamples");


  // Count number of samples per class


  class_size = iftAllocIntArray(Z->nclasses+1);
  for (s=0; s < Z->nsamples; s++){
    class_size[Z->sample[s].truelabel]++;
  }

  // Verify percentage and number of training samples per class

  Z->ntrainsamples = 0;
  for (c=1; c <= Z->nclasses; c++) {
    nsamples = (int)(train_perc*class_size[c]);
    if (nsamples > class_size[c])
      nsamples = class_size[c];
    if (nsamples == 0)
      nsamples = 1;
    Z->ntrainsamples += nsamples;
    if (nsamples <= 0){
      fprintf(stderr,"For class %d\n",c);
      iftError("No available samples","iftSelectSupTrainSamples");
    }
  }

  // Randomly select samples


  for (c=1; c <= Z->nclasses; c++) {
    nsamples = (int)(train_perc*class_size[c]);
    if (nsamples > class_size[c])
      nsamples = class_size[c];
    if (nsamples == 0)
      nsamples = 1;
    sample = iftAllocIntArray(class_size[c]);
    count  = iftAllocIntArray(class_size[c]);
    t=0;
    for (s=0; s < Z->nsamples; s++)
      if (Z->sample[s].truelabel==c){
	sample[t]=s;
	count[t]=100;
	t++;
      }
    t = 0; high = class_size[c]-1;
    while (t < nsamples) {
      i = iftRandomInteger(0,high);
      s = sample[i];
      if (count[i]==0){
	Z->sample[s].status=IFT_TRAIN;
	iftSwitchValues(&sample[i],&sample[high]);
	iftSwitchValues(&count[i],&count[high]);
	t++; high--;
      }else{
	count[i]--;
      }
    }
    free(count);
    free(sample);
  }

  free(class_size);

  return(Z->ntrainsamples);
}

int main(int argc, char *argv[])
{
  
  iftDataSet      *dataset=NULL;
  iftKnnGraph     *graph=NULL;
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
	  
	  printf("nRun: %d\n", j);
  	  iftDataSet      *Z=NULL;
	  iftDataSet      *Test=NULL;
	  iftSelectSupTrainSamples(dataset, 0.80);
	  Z = iftExtractSamples(dataset,IFT_TRAIN);
	  Test = iftExtractSamples(dataset,IFT_TEST);
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
			}else{
				cnt++;
			}
		}
		if(cnt == Z->nclasses)
			  all_cls = 1;
	  }

	  Z->ntrainsamples = Z->nlabels;

	  /*
	  printf("Z nsamples : %d, ntrainsamples : %d \n", Z->nsamples, Z->ntrainsamples);
	  iftDataSet *reducedZ = iftGraphBoundaryReduction(graph, Z);
	  printf("%d samples in the reduced set.\n", reducedZ->nsamples);

	  iftMST *reducedMST = iftCreateMST(reducedZ);

	  for (i = 0; i < reducedZ->nsamples; i++) {
		  reducedZ->sample[i].status = NIL;
		  reducedZ->sample[i].truelabel = NIL;
	  }
	  reducedZ->ntrainsamples = 0;

	  iftSortNodesByWeightMST(reducedMST, DECREASING);
	  */
	  iftCplGraph *trainGraph = iftCreateCplGraph(Z);
	  iftSupTrain(trainGraph);
	  printf("Trained graph with %d root nodes.\n", trainGraph->nnodes);
	  
	  int iteration = 0;
	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;
	  do {
	      tstart = iftTic();
		  printf("Iteration %d\n", iteration+1);
	      tic = iftTic();
		  //iftClassify(trainGraph, reducedZ);

		  //iftSet *selectedSamples = iftSelectSamplesForUserLabelingMST(reducedMST, Z->nclasses * 2); // select 2 * number of classes
	      iftSet *selectedSamples = iftSelectSamplesForUserLabelingMU(trainGraph, Z, Z->nclasses * 2); // select 2 * number of classes
	      toc = iftToc();
	      time_selection[iteration] += iftCompTime(tic,toc) / (float)nrun;

		  int userCorrections = 0;
		  int sampleIdx;
		  int nsamples = 0;
		  while ((sampleIdx = iftRemoveSet(&selectedSamples)) != NIL) {
			  nsamples++;
			  if (Z->sample[sampleIdx].truelabel != Z->sample[sampleIdx].label) { // if sample was misclassified
				  userCorrections++;
			  }
			  Z->sample[sampleIdx].status = IFT_TRAIN;
			  Z->ntrainsamples++;
		  }
		  ntrainsamples[iteration] = (float)Z->ntrainsamples / (float)nrun;
		  printf("# train samples %d.\n", Z->ntrainsamples);
		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
	      num_showed[iteration] += nsamples / (float)nrun;
	      num_annotated[iteration] += userCorrections / (float)nrun;

		  if (nsamples == 0) break;

		  iftDestroyCplGraph(&trainGraph);

		  nSamplesCorrected += userCorrections;
		  nSamplesShown += nsamples;

	      tic = iftTic();
		  trainGraph = iftCreateCplGraph(Z);
		  iftSupTrain(trainGraph);
	      toc = iftToc();
	      time_train[iteration] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Trained graph with %d nodes.\n", trainGraph->nnodes);

	      tic = iftTic();
		  iftClassify(trainGraph, Test);
	      //iftClassify(trainGraph, Z);
		  accuracy_iter = iftTruePositives(Test);
	      //accuracy_iter = iftClassifAccuracy(Z);
		  accuracy[iteration] += (float)accuracy_iter/ (float)nrun;
	      toc = iftToc();
	      time_cls[iteration] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);

		  tfinish = iftToc();
		  time_iter[iteration] += iftCompTime(tstart,tfinish) / (float)nrun;
		  
		  iteration++;
	  } while (iteration < max_iteration);

	  //iftDestroyDataSet(&reducedZ);
	  //iftDestroyMST(&reducedMST);
	  iftDestroyDataSet(&Test);
	  iftDestroyCplGraph(&trainGraph);
	  //iftDestroyDataSet(&Z);
  }

  for (j = 0; j < max_iteration; j++) {
  	fprintf(f,"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", time_selection[j], time_train[j], time_cls[j], accuracy[j], num_showed[j], num_annotated[j], time_iter[j], ntrainsamples[j]);
  }
  fclose(f);

  printf("\n");
  printf("Finish exp. Active Learning OPF\n");

  iftDestroyKnnGraph(&graph);
  iftDestroyDataSet(&dataset);
  

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
