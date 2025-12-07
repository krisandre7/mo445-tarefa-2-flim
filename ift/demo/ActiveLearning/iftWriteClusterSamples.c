#include "ift.h"
#include <assert.h>

/**
 * Active learning demo that uses clustering, boundary reduction, MST and boundary samples.
 **/

int iftSelectRandTrainSamples(iftDataSet *Z, int n)
{
  int nsamples_sel, i, count, s, available, selected;
  nsamples_sel = n;
  available = Z->nsamples - Z->ntrainsamples;
  if (available < n ){
    nsamples_sel = available;
    for (s=0; s < Z->nsamples; s++)
      Z->sample[s].status = IFT_TRAIN;
    Z->ntrainsamples = Z->nsamples;
    return nsamples_sel;
  }

  int *test_arr = iftAllocIntArray(available);
  int *test_sel = iftAllocIntArray(available);
  count = 0;
  for (s=0; s < Z->nsamples; s++) {
    if(Z->sample[s].status != IFT_TRAIN){
      test_arr[count] = s;
      test_sel[count] = 0;
      count++;
    }
  }

  if(count != available)
    iftError("ntrainsamples is not updated","iftSelectRandTrainSamples");

  count = 0;
  while (count < nsamples_sel) {
    i = iftRandomInteger(0,available-1);
    if(test_sel[i] == 0){
      selected = test_arr[i];
      Z->sample[selected].status = IFT_TRAIN;
      test_sel[i] = 1;
      Z->ntrainsamples++;
      count++;
    }
  }
  free(test_arr);
  free(test_sel);
  return nsamples_sel;

}

int* iftSelectRandSamples(int* arr, int len_arr, int n)
{
  int i, count, s;

  if(len_arr < n)
    iftError("wrong n value","iftSelectRandSamples");

  int *val = iftAllocIntArray(n);
  int *sel = iftAllocIntArray(n);
  for (s=0; s < n; s++) {
      sel[s] = 0;
  }
  
  count = 0;
  while (count < n) {
    i = iftRandomInteger(0,len_arr-1);
    if(sel[i] == 0){
      val[count] = arr[i];
      sel[i] = 1;
      count++;
    }
  }
  free(sel);
  return val;

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
    class_size[Z->sample[s].class]++;
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
      if (Z->sample[s].class==c){
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
  
  //iftDataSet      *dataset=NULL;
  iftDataSet      *Z=NULL;
  //iftDataSet      *sortZ=NULL;
  iftKnnGraph     *graph=NULL;
  timer           *t1=NULL,*t2=NULL;
  int i;

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
    iftError("Usage: iftWriteCluster <OPF dataset.dat> <kmax_perc [0,1]> <clusters CSV>","main");

  Z = iftReadOPFDataSet(argv[1]);

  printf("Read dataset with %d samples.\n", Z->nsamples);

  FILE *f1 = fopen(argv[3], "w");
  srand(time(NULL));


  float accuracy_iter;
  
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
	  		cls[Z->sample[sampleIdx].class] = 1;
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
  int *nNodes;
  int **nodeLists = iftGetNodesPerClusterOrderedByDistanceToRoot(graph, &nNodes);
  */

  
  printf("Z nsamples : %d, ntrainsamples : %d \n", Z->nsamples, Z->ntrainsamples);
  iftDataSet *reducedZ = iftGraphBoundaryReduction(graph, Z);
  printf("%d samples in the reduced set.\n", reducedZ->nsamples);
  
  /*
  iftMST *reducedMST = iftCreateMST(reducedZ);

  for (i = 0; i < reducedZ->nsamples; i++) {
	  reducedZ->sample[i].status = NIL;
	  reducedZ->sample[i].class = NIL;
  }
  reducedZ->ntrainsamples = 0;

  iftSortNodesByWeightMST(reducedMST, DECREASING);
  //iftSortNodesByWeightMST(reducedMST, INCREASING);
  */
  


  // Write index clusters
  int numsamples = 0;
  for (i = 0; i < Z->nsamples; i++) {
	  if(Z->sample[i].status == IFT_TRAIN){
		  fprintf(f1,"%d\n", i);
                  numsamples++;
	  }
  }
 
  // write random boundary samples
  iftSet *bsamples = iftGetKnnBoundarySamples(graph);
  //int max_bsamples = 30;
  int max_bsamples = 25;
  int *arr_bsamples = iftAllocIntArray(reducedZ->nsamples);
  int count = 0;
  int sampleIdx;
  while ((sampleIdx = iftRemoveSet(&bsamples)) != NIL) {
  	arr_bsamples[count] = sampleIdx;
  	count++;
  }
  int *sel_bsamples = iftSelectRandSamples(arr_bsamples, reducedZ->nsamples, max_bsamples);
  for (i = 0; i < max_bsamples; i++) {
	  fprintf(f1,"%d\n", sel_bsamples[i]);
          numsamples++;
  }
  /*
  // write the nearest samples to the centers
  int maxsamples_perlabel = 2;
  for (i = 1; i <= Z->nlabels; i++) {
	  int numperlabel = 0;
	  for (j = 1; j <= maxsamples_perlabel; j++) {
		  if(j < nNodes[i]){
			  fprintf(f1,"%d\n", nodeLists[i][j]);
			  numsamples++;
			  numperlabel++;
		  }
	  }
	  printf("label %d: %d \n", i ,numperlabel);
  }
  */
  
  // Write boundary samples
  /*
  int nbsamples_perlabel = 5;
  int nbsamples = 0;
  int *selected = iftAllocIntArray(Z->nlabels);
  for (i = 0; i < Z->nlabels; i++) {
	  selected[i] = 0;
  }
  int u;
  for (u = 0; u < reducedMST->nnodes; u++) {
	  int s = reducedMST->node[u].sample;
	  if(selected[Z->sample[reducedZ->sample[s].id].label-1] < nbsamples_perlabel){
		  fprintf(f1,"%d\n", reducedZ->sample[s].id);
		  selected[Z->sample[reducedZ->sample[s].id].label-1]++;
		  nbsamples++;
		  numsamples++;
	  }
	  if(nbsamples == nbsamples_perlabel*reducedZ->nlabels){
		  break;
	  }
  }
  printf("%d boundary samples\n",nbsamples);
  */
  printf("%d samples selected\n",numsamples);


  fclose(f1);

  iftCplGraph *trainGraph = iftCreateCplGraph(Z);
  iftSupTrain(trainGraph);
  printf("Trained graph with %d root nodes.\n", trainGraph->nnodes);

  iftClassify(trainGraph, Z);
  accuracy_iter = iftTruePositives(Z);
  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);




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
