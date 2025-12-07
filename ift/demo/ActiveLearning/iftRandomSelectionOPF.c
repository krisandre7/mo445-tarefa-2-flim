#include "ift.h"
#include <assert.h>

/**
 * Random selection using OPF
 **/

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

iftSet *iftSelectRandSamplesForUserLabeling(iftDataSet *Z, int n) {
    iftSample *sample = Z->sample;
  iftSet *selectedNodes = NULL;
  
  int nsamples_sel, i, count, s, available, selected, setSize;
  setSize = 0;
  nsamples_sel = n;
  
  available = 0;
  for (s=0; s < Z->nsamples; s++) {
    if (sample[s].status != IFT_TRAIN) {
      available++;
    }
  }
  
  if (available < n ){
    nsamples_sel = available;
    for (s=0; s < Z->nsamples; s++) {
      if (sample[s].status != IFT_TRAIN) { // if not labeled by the user
         setSize += iftUnionSetElem(&selectedNodes, s);
      }
    }
    return selectedNodes;
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

  count = 0;
  while (count < nsamples_sel) {
    i = iftRandomInteger(0,available-1);
    if(test_sel[i] == 0){
      selected = test_arr[i];
      test_sel[i] = 1;
      setSize += iftUnionSetElem(&selectedNodes, selected);
      count++;
    }
  }
  free(test_arr);
  free(test_sel);
  
  return selectedNodes;
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

  iftCplGraph *trainGraph;
  
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

	  t1 = iftTic();
	  iftSelectUnsupTrainSamples(Z,1.0);
	  graph = iftUnsupLearn(Z, atof(argv[2]), iftNormalizedCut, 10);
	  iftUnsupClassify(graph, Z);

	  t2 = iftToc();
	  fprintf(stdout,"clustering in %f ms with %d groups\n",iftCompTime(t1,t2), Z->nlabels);
	  printf("%d nodes in the clustered graph (k=%d kmax=%d).\n", graph->nnodes, graph->k, graph->kmax);    
	  for (i = 0; i < Z->nsamples; i++) {
		  Z->sample[i].status = NIL;
	  }
	  Z->ntrainsamples = 0;
          
	  /* Get same amount of initial training samples */
	  //iftSelectRandTrainSamples(Z, Z->nlabels);
	  float trainPerc = (float) Z->nlabels / (float)Z->nsamples;
	  iftSelectSupTrainSamples1(Z, trainPerc);
      printf("***Z nsamples: %d, ntrainsamples: %d, trainPerc: %f \n", Z->nsamples, Z->ntrainsamples, trainPerc);
 
	  int iteration = 0;
	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;
          
	  do {
		  tstart = iftTic();
		  printf("Iteration %d\n", iteration+1);
		  tic = iftTic();
		  
		  iftSet *selectedSamples = iftSelectRandSamplesForUserLabeling(Z, Z->nclasses * 2); // select 2 * number of classes
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
		  num_showed[iteration] += (float)nsamples/ (float)nrun;
		  num_annotated[iteration] += (float)userCorrections/ (float)nrun;

		  if (nsamples == 0) break;

		  nSamplesCorrected += userCorrections;
		  nSamplesShown += nsamples;

		  tic = iftTic();
		  trainGraph = iftCreateCplGraph(Z);
		  iftSupTrain(trainGraph);
		  toc = iftToc();
		  time_train[iteration] += iftCompTime(tic,toc)/ (float)nrun;
		  printf("Trained graph with %d nodes.\n", trainGraph->nnodes);

		  tic = iftTic();
		  iftClassify(trainGraph, Test);
          accuracy_iter = iftTruePositives(Test);
		  accuracy[iteration] += accuracy_iter / (float)nrun;
		  toc = iftToc();
		  time_cls[iteration] += iftCompTime(tic,toc)/ (float)nrun;
		  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);

		  tfinish = iftToc();
		  time_iter[iteration] += iftCompTime(tstart,tfinish)/ (float)nrun;
                  iteration++;

		  printf("\n");
		  printf("%d samples shown to the user.\n", nSamplesShown);
		  printf("%d samples corrected by the user.\n", nSamplesCorrected);
                  
	  } while (iteration < max_iteration);
	  
	  iftDestroyDataSet(&Test);
	  iftDestroyDataSet(&Z);
  }
  
  for (j = 0; j < max_iteration; j++) {
  	fprintf(f,"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", time_selection[j], time_train[j], time_cls[j], accuracy[j], num_showed[j], num_annotated[j], time_iter[j], ntrainsamples[j]);
  }
  
  fclose(f);

  printf("\n");
  printf("Finish exp. Random Selection OPF\n");

  iftDestroyCplGraph(&trainGraph);
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
