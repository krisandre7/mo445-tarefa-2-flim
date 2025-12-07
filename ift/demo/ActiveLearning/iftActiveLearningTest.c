#include "ift.h"
#include <assert.h>

/**
 * Active learning demo that uses clustering, boundary reduction, MST and boundary samples.
 **/

typedef struct iftSortSample {
	int sampleIndex;
	float weight;
	int label1;
	int label2;
} iftSortSmaple;

static int cmpSortSampleDescending(  void *p1,   void *p2) {
	  float weight1 = ((iftSortSmaple *) p1)->weight;
	  float weight2 = ((iftSortSmaple *) p2)->weight;
	if (weight1 < weight2) {
		return 1;
	} else if (weight1 > weight2) {
		return -1;
	}
	return 0;
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

iftSet *  iftSelectSamplesForUserLabelingMU_old(iftCplGraph *graph, iftDataSet *Z, int n)
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

#pragma omp parallel for schedule(dynamic) shared(Z,Z1,graph,iftDist,setSize,selectedNodes) private(t,i,u,s,weight,minCost,minCost2,label,label2,tmp)
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

iftSet *  iftSelectSamplesForUserLabelingMU1(iftCplGraph *graph, iftDataSet *Z, int n)
{
	iftSet *selectedNodes = NULL;
	int u, s, t, i, setSize, maxSortSamples, numSortSamples, label = 0,label2 = 0;
	float tmp, weight, minCost, minCost2;
	iftDataSet *Z1=graph->Z;
	maxSortSamples = 1000;
	numSortSamples = 0;
	setSize = 0;
	iftSortSmaple *sortSamples = (iftSortSmaple *) malloc(sizeof(iftSortSmaple) * maxSortSamples);

	// Init DataSet
	for(i=0; i< maxSortSamples; i++){
		sortSamples[i].sampleIndex = -1;
		sortSamples[i].weight = -1.0;
	}

	// Check Dataset
	if (Z1->nfeats != Z->nfeats)
	  {
	    char msg[200];
	    sprintf(msg,"Incompatible datasets (%d/%d)",Z1->nfeats,Z->nfeats);
	    iftError(msg,"iftClassify");
	  }

#pragma omp parallel for schedule(dynamic) shared(Z,Z1,graph,iftDist,numSortSamples) private(t,i,u,s,weight,minCost,minCost2,label,label2,tmp)
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
			if(ratio > 0.5){
				//printf("t: %d, ratio: %f, minCost: %f %d, minCost2: %f %d\n", t, ratio, minCost, label, minCost2, label2);
				if(numSortSamples < maxSortSamples){
					sortSamples[numSortSamples].sampleIndex = t;
					sortSamples[numSortSamples].weight = ratio;
					numSortSamples++;
				}
			}
		}
	}

	// Decreasing order
	qsort(sortSamples, numSortSamples, sizeof(iftSortSmaple), cmpSortSampleDescending);
	for(i=0; i< numSortSamples; i++){
		printf("index: %d, ratio: %f\n", sortSamples[i].sampleIndex, sortSamples[i].weight);
		if(setSize < n) {
			setSize += iftUnionSetElem(&selectedNodes, sortSamples[i].sampleIndex);
		}else{
			break;
		}
	}

	return(selectedNodes);
}


iftSet *  iftSelectSamplesForUserLabelingMU(iftCplGraph *graph, iftDataSet *Z, int n)
{
	iftSet *selectedNodes = NULL;
	int u, s, t, i, setSize, maxSortSamples, numSortSamples, label = 0,label2 = 0;
	float tmp, weight, minCost, minCost2;
	int *selectedClasses;
	iftDataSet *Z1=graph->Z;
	maxSortSamples = 1000;
	numSortSamples = 0;
	setSize = 0;
	iftSortSmaple *sortSamples = (iftSortSmaple *) malloc(sizeof(iftSortSmaple) * maxSortSamples);

	// Init DataSet
	for(i=0; i< maxSortSamples; i++){
		sortSamples[i].sampleIndex = -1;
		sortSamples[i].weight = -1.0;
	}

	selectedClasses = iftAllocIntArray(Z1->nclasses*Z1->nclasses);
	for(i=0; i< Z1->nclasses*Z1->nclasses; i++){
		selectedClasses[i] = 0;
	}

	// Check Dataset
	if (Z1->nfeats != Z->nfeats)
	  {
	    char msg[200];
	    sprintf(msg,"Incompatible datasets (%d/%d)",Z1->nfeats,Z->nfeats);
	    iftError(msg,"iftClassify");
	  }

#pragma omp parallel for schedule(dynamic) shared(Z,Z1,graph,iftDist,numSortSamples,selectedClasses) private(t,i,u,s,weight,minCost,minCost2,label,label2,tmp)
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
			if(ratio > 0.5){
				//printf("t: %d, ratio: %f, minCost: %f %d, minCost2: %f %d\n", t, ratio, minCost, label, minCost2, label2);
				if(numSortSamples < maxSortSamples && selectedClasses[(label-1)*Z1->nclasses+(label2-1)] == 0){
					sortSamples[numSortSamples].sampleIndex = t;
					sortSamples[numSortSamples].weight = ratio;
					sortSamples[numSortSamples].label1 = label;
					sortSamples[numSortSamples].label2 = label2;
					selectedClasses[(label-1)*Z1->nclasses+(label2-1)] = 1;
					//selectedClasses[(label2-1)*Z1->nclasses+(label-1)] = 1;
					numSortSamples++;
				}
			}
		}
	}

	// Decreasing order
	qsort(sortSamples, numSortSamples, sizeof(iftSortSmaple), cmpSortSampleDescending);
	for(i=0; i< numSortSamples; i++){
		printf("index: %d, ratio: %f, label: %d, label2: %d\n", sortSamples[i].sampleIndex, sortSamples[i].weight, sortSamples[i].label1, sortSamples[i].label2);
		if(setSize < n) {
			setSize += iftUnionSetElem(&selectedNodes, sortSamples[i].sampleIndex);
		}else{
			break;
		}
	}
	free(selectedClasses);
	return(selectedNodes);
}

float* iftGridSearch(iftDataSet* Z, float percTrainVal, int nRUN){

	iftDataSet *Ztrain[2], *Ztest[2],*Ztmp;
	iftSVM      *svm   = NULL;
	int i,j,it;
	float C, sigma;
	float max_acc = 0;
	float *values;
	values = iftAllocFloatArray(2);
	float sigmas[5], Cs[5];
	sigmas[0] = 0.010000;
	sigmas[1] = 0.056234;
	sigmas[2] = 0.316228;
	sigmas[3] = 1.778279;
	sigmas[4] = 10.000000;
	Cs[0] = 1;
	Cs[1] = 10;
	Cs[2] = 100;
	Cs[3] = 1000;
	Cs[4] = 10000;
	C = 100000;
	sigma = 0.1;
	for(i=0; i<5; i++){
		for(j=0; j<5; j++){
			float acc = 0;
			for(it=0;it<nRUN;it++){
				Ztmp = iftCopyDataSet(Z, true);
				iftSetStatus(Ztmp, IFT_TEST);
				iftSelectSupTrainSamples(Ztmp,percTrainVal); // Selecting training samples
				/* Normalize Train and Test */
				Ztrain[0] = iftExtractSamples(Ztmp, IFT_TRAIN);
				Ztest[0] = iftExtractSamples(Ztmp, IFT_TEST);

				Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
				iftDestroyDataSet(&Ztrain[0]);
				Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
				iftDestroyDataSet(&Ztest [0]);

				/* Train RBF Kernel */
				svm = iftCreateRBFSVC(Cs[i], sigmas[j]);
				iftSVMTrainOVO(svm,Ztrain[1]);

				/* Classify RBF Kernel */
				iftSVMClassifyOVO(svm, Ztest[1], IFT_TEST); // Classification
				acc += iftTruePositives(Ztest[1]) / (float) nRUN;

				iftDestroyDataSet(&Ztrain[1]);
				iftDestroyDataSet(&Ztest[1]);

				iftDestroySVM(svm);
				iftDestroyDataSet(&Ztmp);

			}
			//printf("s=%f, C=%f mean acc: %f\n", Cs[i], sigmas[j], acc);
			if(acc > max_acc){
				max_acc = acc;
				C = Cs[i];
				sigma = sigmas[j];
			}
		}
	}
	values[0] = C;
	values[1] = sigma;
	return values;
}

float *iftAL_OPF_MU(iftDataSet *Z, iftDataSet *Test, float kval, int max_iteration, int nrun){
	  float *values;
	  int i;
	  timer *t1=NULL,*t2=NULL;
	  iftKnnGraph *graph = NULL;

	  int ncols = 8;
	  float accuracy_iter;
	  float k = kval;


	  int all_cls = 0;

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

		// check if the graph has roots from all classes
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


	  //float trainPerc = (float) Z->nlabels / (float)Z->nsamples;
	  //iftSelectSupTrainSamples1(Z, trainPerc);

	  timer *tstart, *tfinish;
	  timer *tic,*toc;
	  int iteration = 0;

	  values = iftAllocFloatArray(max_iteration*ncols+1);
	  values[max_iteration*ncols] = Z->ntrainsamples;

	  printf("Iteration %d\n", iteration);
	  iftCplGraph *trainGraph = iftCreateCplGraph(Z);
	  iftSupTrain(trainGraph);
	  printf("Trained graph with %d root nodes.\n", trainGraph->nnodes);


	  iftClassify(trainGraph, Test);
	  accuracy_iter = iftTruePositives(Test);
	  values[iteration * ncols + 5] += (float)accuracy_iter/ (float)nrun;
	  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
	  iteration++;

	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;
	  do {
		  tstart = iftTic();
		  printf("Iteration %d\n", iteration);
		  tic = iftTic();
		  //iftClassify(trainGraph, reducedZ);

		  iftSet *selectedSamples = iftSelectSamplesForUserLabelingMU(trainGraph, Z, Z->nclasses * 2); // select 2 * number of classes
		  if(selectedSamples == NULL)
		  	  selectedSamples = iftSelectRandSamplesForUserLabeling(Z, Z->nclasses * 2);

		  toc = iftToc();
		  values[iteration * ncols + 0] += iftCompTime(tic,toc) / (float)nrun;

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
		  values[iteration * ncols + 1] = (float)Z->ntrainsamples / (float)nrun;
		  printf("# train samples %d.\n", Z->ntrainsamples);
		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
		  values[iteration * ncols + 2] += nsamples / (float)nrun;
		  values[iteration * ncols + 3] += userCorrections / (float)nrun;

		  //if (nsamples == 0) break;

		  iftDestroyCplGraph(&trainGraph);

		  nSamplesCorrected += userCorrections;
		  nSamplesShown += nsamples;

		  tic = iftTic();
		  trainGraph = iftCreateCplGraph(Z);
		  iftSupTrain(trainGraph);
		  toc = iftToc();
		  values[iteration * ncols + 4] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Trained graph with %d nodes.\n", trainGraph->nnodes);

		  tic = iftTic();
		  iftClassify(trainGraph, Test);
		  //iftClassify(trainGraph, Z);
		  accuracy_iter = iftTruePositives(Test);
		  //accuracy_iter = iftClassifAccuracy(Z);
		  values[iteration * ncols + 5] += (float)accuracy_iter/ (float)nrun;
		  toc = iftToc();
		  values[iteration * ncols + 6] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Classification accuracy in the test dataset: %.2f%%\n", accuracy_iter * 100.0f);

		  tfinish = iftToc();
		  values[iteration * ncols + 7] += iftCompTime(tstart,tfinish) / (float)nrun;

		  iteration++;
	  } while (iteration < max_iteration);

	  iftDestroyCplGraph(&trainGraph);
	  iftDestroyKnnGraph(&graph);

	  return values;

}


float *iftAL_OPF_RDS(iftDataSet *Z, iftDataSet *Test, float kval, int max_iteration, int nrun){
	  float *values;
	  int i;
	  timer *t1=NULL,*t2=NULL;
	  iftKnnGraph *graph = NULL;

	  int ncols = 8;
	  float accuracy_iter;
	  float k = kval;

	  int all_cls = 0;
	  // Clustering
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

		// check if the graph has roots from all classes
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
	  timer *tstart, *tfinish;
	  timer *tic,*toc;
	  int iteration = 0;

	  values = iftAllocFloatArray(max_iteration*ncols+1);
	  values[max_iteration*ncols] = Z->ntrainsamples;

	  int *nNodes;
	  int **nodeLists = iftGetNodesPerClusterOrderedByDistanceToRoot(graph, &nNodes);

	  printf("Iteration %d\n", iteration);
	  // create train graph with root nodes (status = IFT_TRAIN)
	  iftCplGraph *trainGraph = iftCreateCplGraph(Z);
	  iftSupTrain(trainGraph);
	  printf("Trained graph with %d root nodes.\n", trainGraph->nnodes);

	  iftClassify(trainGraph, Test);
	  accuracy_iter = iftTruePositives(Test);
	  values[iteration * ncols + 5] += (float)accuracy_iter/ (float)nrun;
	  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
	  iteration++;


	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;
	  do {
		  tstart = iftTic();
		  printf("Iteration %d\n", iteration);
		  tic = iftTic();
		  // select at most (2 * number of classes) samples without IFT_TRAIN status
		  iftSet *selectedSamples = iftGetRootDistanceBasedSamples(trainGraph, Z, nodeLists, nNodes, IFT_TRAIN, 1, Z->nclasses * 2);
		  if(selectedSamples == NULL)
		      selectedSamples = iftSelectRandSamplesForUserLabeling(Z, Z->nclasses * 2);

		  toc = iftToc();
		  values[iteration * ncols + 0] += iftCompTime(tic,toc) / (float)nrun;
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
		  values[iteration * ncols + 1] += (float)Z->ntrainsamples / (float)nrun;
		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
		  values[iteration * ncols + 2] += nsamples / (float)nrun;
		  values[iteration * ncols + 3] += userCorrections / (float)nrun;

		  //if (nsamples == 0) break;

		  iftDestroyCplGraph(&trainGraph);

		  nSamplesCorrected += userCorrections;
		  nSamplesShown += nsamples;

		  tic = iftTic();
		  trainGraph = iftCreateCplGraph(Z);
		  iftSupTrain(trainGraph);
		  toc = iftToc();
		  values[iteration * ncols + 4] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Trained graph with %d nodes.\n", trainGraph->nnodes);

		  tic = iftTic();
		  iftClassify(trainGraph, Test);
		  accuracy_iter = iftTruePositives(Test);
		  values[iteration * ncols + 5] = accuracy_iter / (float)nrun;
		  toc = iftToc();
		  values[iteration * ncols + 6] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Classification accuracy in the test dataset: %.2f%%\n", accuracy_iter * 100.0f);
		  printf("\n");

		  tfinish = iftToc();
		  values[iteration * ncols + 7] = iftCompTime(tstart,tfinish) / (float)nrun;

		  iteration++;
	  } while (iteration < max_iteration);

	  free(nNodes);
	  for (i = 1; i <= Z->nlabels; i++) {
		  free(nodeLists[i]);
	  }
	  free(nodeLists);

	  iftDestroyCplGraph(&trainGraph);
	  iftDestroyKnnGraph(&graph);

	  return values;
}

float *iftAL_OPF(iftDataSet *Z, iftDataSet *Test, float kval, int max_iteration, int nrun){
	  float *values;
	  int i;
	  timer *t1=NULL,*t2=NULL;
	  iftKnnGraph *graph = NULL;

	  int ncols = 8;
	  float accuracy_iter;
	  float k = kval;

	  int all_cls = 0;
	  // Clustering
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

		// check if the graph has roots from all classes
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


	  timer *tstart, *tfinish;
      timer *tic,*toc;
      int iteration = 0;

	  values = iftAllocFloatArray(max_iteration*ncols+1);
	  values[max_iteration*ncols] = Z->ntrainsamples;

	  iftDataSet *reducedZ = iftGraphBoundaryReduction(graph, Z);
	  printf("%d samples in the reduced set.\n", reducedZ->nsamples);

	  iftMST *reducedMST = iftCreateMST(reducedZ);

	  for (i = 0; i < reducedZ->nsamples; i++) {
		  reducedZ->sample[i].status = NIL;
		  reducedZ->sample[i].truelabel = NIL;
	  }
	  reducedZ->ntrainsamples = 0;

	  iftSortNodesByWeightMST(reducedMST, DECREASING);

	  printf("Iteration %d\n", iteration);
	  iftCplGraph *trainGraph = iftCreateCplGraph(Z);
	  iftSupTrain(trainGraph);
	  printf("Trained graph with %d root nodes.\n", trainGraph->nnodes);
	  
	  iftClassify(trainGraph, Test);
	  accuracy_iter = iftTruePositives(Test);
	  values[iteration * ncols + 5] += (float)accuracy_iter/ (float)nrun;
	  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
	  iteration++;


	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;
	  do {
		  tstart = iftTic();
		  printf("Iteration %d\n", iteration);
		  tic = iftTic();
		  iftClassify(trainGraph, reducedZ);

		  iftSet *selectedSamples = iftSelectSamplesForUserLabelingMST(reducedMST, Z->nclasses * 2); // select 2 * number of classes
		  if(selectedSamples == NULL)
		  	  selectedSamples = iftSelectRandSamplesForUserLabeling(reducedZ, Z->nclasses * 2);

		  toc = iftToc();
		  values[iteration * ncols + 0] += iftCompTime(tic,toc) / (float)nrun;

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
		  values[iteration * ncols + 1] += (float)Z->ntrainsamples / (float)nrun;
		  printf("# train samples %d.\n", Z->ntrainsamples);
		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
		  values[iteration * ncols + 2] += nsamples / (float)nrun;
		  values[iteration * ncols + 3] += userCorrections / (float)nrun;

		  //if (nsamples == 0) break;

		  iftDestroyCplGraph(&trainGraph);

		  nSamplesCorrected += userCorrections;
		  nSamplesShown += nsamples;

		  tic = iftTic();
		  trainGraph = iftCreateCplGraph(Z);
		  iftSupTrain(trainGraph);
		  toc = iftToc();
		  values[iteration * ncols + 4] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Trained graph with %d nodes.\n", trainGraph->nnodes);

		  tic = iftTic();
		  iftClassify(trainGraph, Test);
		  accuracy_iter = iftTruePositives(Test);
		  values[iteration * ncols + 5] = accuracy_iter / (float)nrun;
		  toc = iftToc();
		  values[iteration * ncols + 6] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Classification accuracy in the test dataset: %.2f%%\n", accuracy_iter * 100.0f);

		  tfinish = iftToc();
		  values[iteration * ncols + 7] = iftCompTime(tstart,tfinish) / (float)nrun;
		  
		  iteration++;
	  } while (iteration < max_iteration);

	  iftDestroyDataSet(&reducedZ);
	  iftDestroyMST(&reducedMST);
	  iftDestroyCplGraph(&trainGraph);

	  iftDestroyKnnGraph(&graph);

	  return values;
}


float *iftRS_OPF(iftDataSet *Z, iftDataSet *Test, float kval, int max_iteration, int nrun){
	  timer *tstart, *tfinish;
	  timer *tic,*toc;
	  float accuracy_iter;
	  float *values;
	  int ncols = 8;
	  iftCplGraph *trainGraph;
	  int iteration = 0;

	  values = iftAllocFloatArray(max_iteration*ncols);
	  float trainPerc = (float) Z->nlabels / (float)Z->nsamples;

	  iftSelectSupTrainSamples1(Z, trainPerc);
	  printf("*** Z nsamples: %d, ntrainsamples: %d, init_ntrain: %d, trainPerc:%f\n",Z->nsamples,Z->ntrainsamples,Z->nlabels,trainPerc);

	  printf("Iteration %d\n", iteration);
	  trainGraph = iftCreateCplGraph(Z);
	  iftSupTrain(trainGraph);
	  printf("Trained graph with %d root nodes.\n", trainGraph->nnodes);

	  iftClassify(trainGraph, Test);
	  accuracy_iter = iftTruePositives(Test);
	  values[iteration * ncols + 5] += (float)accuracy_iter/ (float)nrun;
	  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
	  iteration++;


	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;

	  do {
		  tstart = iftTic();
		  printf("Iteration %d\n", iteration);
		  tic = iftTic();

		  iftSet *selectedSamples = iftSelectRandSamplesForUserLabeling(Z, Z->nclasses * 2); // select 2 * number of classes
		  toc = iftToc();
		  values[iteration * ncols + 0] += iftCompTime(tic,toc) / (float)nrun;

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
		  values[iteration * ncols + 1] = (float)Z->ntrainsamples / (float)nrun;
		  printf("# train samples %d.\n", Z->ntrainsamples);
		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
		  values[iteration * ncols + 2] += (float)nsamples/ (float)nrun;
		  values[iteration * ncols + 3] += (float)userCorrections/ (float)nrun;

		  //if (nsamples == 0) break;

		  nSamplesCorrected += userCorrections;
		  nSamplesShown += nsamples;

		  tic = iftTic();
		  trainGraph = iftCreateCplGraph(Z);
		  iftSupTrain(trainGraph);
		  toc = iftToc();
		  values[iteration * ncols + 4] += iftCompTime(tic,toc)/ (float)nrun;
		  printf("Trained graph with %d nodes.\n", trainGraph->nnodes);

		  tic = iftTic();
		  iftClassify(trainGraph, Test);
		  accuracy_iter = iftTruePositives(Test);
		  values[iteration * ncols + 5] += accuracy_iter / (float)nrun;
		  toc = iftToc();
		  values[iteration * ncols + 6] += iftCompTime(tic,toc)/ (float)nrun;
		  printf("Classification accuracy in the test dataset: %.2f%%\n", accuracy_iter * 100.0f);

		  tfinish = iftToc();
		  values[iteration * ncols + 7] += iftCompTime(tstart,tfinish)/ (float)nrun;
				  iteration++;

		  //printf("\n");
		  //printf("%d samples shown to the user.\n", nSamplesShown);
		  //printf("%d samples corrected by the user.\n", nSamplesCorrected);

	  } while (iteration < max_iteration);


	  return values;
}

float *iftAL_SVM(iftDataSet *Z, iftDataSet *Test, float kval, int max_iteration, int nrun){

	  timer *tstart, *tfinish;
	  timer *tic,*toc;
	  timer *t1=NULL,*t2=NULL;
	  iftKnnGraph *graph = NULL;
	  float accuracy_iter;
	  float *values;
	  int ncols = 8;
	  int i,s;
	  float k = kval;
	  values = iftAllocFloatArray(max_iteration*ncols+1);

	  int all_cls = 0;
	  /* Clustering */
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
	  values[max_iteration*ncols] = Z->ntrainsamples;

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
	  int iteration = 0;
	  float cSVM = 1.0;
	  iftSVM      *svm   = NULL;
	  iftDataSet *Ztrain[2], *Ztest[2], *Ztmp;

	  printf("Iteration %d\n", iteration);
	  tic = iftTic();
	  Ztmp = iftCopyDataSet(Z, true);
	  printf("ntrainsamples %d\n",Ztmp->ntrainsamples);
	  Ztrain[0] = iftExtractSamples(Ztmp,IFT_TRAIN);
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
	  values[iteration * ncols + 4] = iftCompTime(tic,toc) / (float)nrun;

	  /* Classify Test */
	  tic = iftTic();
	  iftSVMLinearClassifyOVA(svm, Ztest[1], Ztrain[1], IFT_TEST); // Classification
	  accuracy_iter = iftTruePositives(Ztest[1]);
	  values[iteration * ncols + 5] += accuracy_iter / (float)nrun;
	  toc = iftToc();
	  values[iteration * ncols + 6] += iftCompTime(tic,toc) / (float)nrun;
	  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
	  iteration++;

	  iftDestroyDataSet(&Ztest[1]);
	  iftDestroySVM(svm);



	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;
	  do {
		  tstart = iftTic();
		  printf("Iteration %d\n", iteration);

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
		  if(selectedSamples == NULL)
		       selectedSamples = iftSelectRandSamplesForUserLabeling(reducedZ, Z->nclasses * 2);
		  toc = iftToc();
		  values[iteration * ncols + 0] += iftCompTime(tic,toc) / (float)nrun;

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
		  values[iteration * ncols + 1] = (float)Z->ntrainsamples / (float)nrun;
		  printf("# train samples %d.\n", Z->ntrainsamples);

		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
		  values[iteration * ncols + 2] += nsamples / (float)nrun;
		  values[iteration * ncols + 3] += userCorrections / (float)nrun;

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
		  values[iteration * ncols + 4] += iftCompTime(tic,toc) / (float)nrun;

		  /* Classify Z */
		  tic = iftTic();
		  iftSVMLinearClassifyOVA(svm, Ztest[1], Ztrain[1], IFT_TEST); // Classification
		  accuracy_iter = iftTruePositives(Ztest[1]);
		  values[iteration * ncols + 5] += accuracy_iter  / (float)nrun;
		  toc = iftToc();
		  values[iteration * ncols + 6] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
		  tfinish = iftToc();
		  values[iteration * ncols + 7] += iftCompTime(tstart,tfinish) / (float)nrun;

		  iftDestroyDataSet(&Ztest[1]);
		  iftDestroySVM(svm);

		  iteration++;
	  } while (iteration < max_iteration);
	  printf("Finish AL SVM\n");
	  iftDestroyDataSet(&reducedZ);
	  iftDestroyMST(&reducedMST);

	  iftDestroyKnnGraph(&graph);

	  return values;
}


float *iftRS_SVM(iftDataSet *Z, iftDataSet *Test, float kval, int max_iteration, int nrun){
	  timer *tstart, *tfinish;
	  timer *tic,*toc;
	  float accuracy_iter;
	  float *values;
	  int ncols = 8;
	  values = iftAllocFloatArray(max_iteration*ncols);

      /* Get same amount of initial training samples */
	  float trainPerc = (float) Z->nlabels / (float)Z->nsamples;
	  iftSelectSupTrainSamples1(Z, trainPerc);
	  //iftSelectRandTrainSamples(Z, Z->nlabels);
	  printf("*** Z nsamples: %d, ntrainsamples: %d, init_ntrain: %d, trainPerc:%f\n",Z->nsamples,Z->ntrainsamples,Z->nlabels,trainPerc);

	  int iteration = 0;
	  float cSVM = 1.0;
	  iftSVM      *svm   = NULL;
	  iftDataSet *Ztrain[2], *Ztest[2], *Ztmp;

	  printf("Iteration %d\n", iteration);
	  /* Train */
	  tic = iftTic();
	  Ztmp = iftCopyDataSet(Z, true);
	  Ztrain[0] = iftExtractSamples(Ztmp,IFT_TRAIN);
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
	  values[iteration * ncols + 4] = iftCompTime(tic,toc) / (float)nrun;

	  /* Classify Test */
	  tic = iftTic();
	  iftSVMLinearClassifyOVA(svm, Ztest[1], Ztrain[1], IFT_TEST); // Classification
	  accuracy_iter = iftTruePositives(Ztest[1]);
	  values[iteration * ncols + 5] += accuracy_iter / (float)nrun;
	  toc = iftToc();
	  values[iteration * ncols + 6] += iftCompTime(tic,toc) / (float)nrun;
	  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
	  iteration++;



	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;
	  do {
		  tstart = iftTic();
		  printf("Iteration %d\n", iteration);

		  tic = iftTic();
		  /* Select samples for user labeling */
		  iftSet *selectedSamples = iftSelectRandSamplesForUserLabeling(Z, Z->nclasses * 2); // select 2 * number of classes
		  toc = iftToc();
		  values[iteration * ncols + 0] += iftCompTime(tic,toc)/ (float)nrun;

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
		  values[iteration * ncols + 1] = (float)Z->ntrainsamples / (float)nrun;
		  printf("# train samples %d.\n", Z->ntrainsamples);
		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
		  values[iteration * ncols + 2] += (float)nsamples/ (float)nrun;
		  values[iteration * ncols + 3] += (float)userCorrections/ (float)nrun;

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
		  values[iteration * ncols + 4] = iftCompTime(tic,toc) / (float)nrun;

		  /* Classify Z */
		  tic = iftTic();
		  iftSVMLinearClassifyOVA(svm, Ztest[1], Ztrain[1], IFT_TEST); // Classification
		  accuracy_iter = iftTruePositives(Ztest[1]);
		  values[iteration * ncols + 5] += accuracy_iter / (float)nrun;
		  toc = iftToc();
		  values[iteration * ncols + 6] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Classification accuracy in the test dataset: %.2f%%\n", accuracy_iter * 100.0f);
		  tfinish = iftToc();
		  values[iteration * ncols + 7] += iftCompTime(tstart,tfinish) / (float)nrun;

		  printf("\n");

		  iftDestroyDataSet(&Ztrain[1]);
		  iftDestroyDataSet(&Ztest[1]);
		  iftDestroySVM(svm);

		  iteration++;

	  } while (iteration < max_iteration);
	  return values;
}


float *iftAL_SVM_RBF(iftDataSet *Z, iftDataSet *Test, float kval, int max_iteration, int nrun){

	  timer *tstart, *tfinish;
	  timer *tic,*toc;
	  timer *t1=NULL,*t2=NULL;
	  iftKnnGraph *graph = NULL;
	  float accuracy_iter;
	  float *values;
	  int ncols = 8;
	  int i,s;
	  float C,sigma;
	  float percTrainVal = 0.33;
	  int nRUNVal = 8;
	  float k = kval;
	  values = iftAllocFloatArray(max_iteration*ncols);

	  int all_cls = 0;
	  /* Clustering */
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

	  iftDataSet *reducedZ = iftGraphBoundaryReduction(graph, Z);
	  printf("%d samples in the reduced set.\n", reducedZ->nsamples);

	  iftMST *reducedMST = iftCreateMST(reducedZ);
	  for (i = 0; i < reducedZ->nsamples; i++) {
		  reducedZ->sample[i].status = NIL;
		  reducedZ->sample[i].truelabel = NIL;
	  }
	  reducedZ->ntrainsamples = 0;

	  iftSortNodesByWeightMST(reducedMST, DECREASING);

	  iftSVM      *svm   = NULL;
	  iftDataSet *Ztrain[2], *Ztest[2], *Ztmp;

	  Ztmp = iftCopyDataSet(Z, true);
	  printf("ntrainsamples %d\n",Ztmp->ntrainsamples);

	  /* Select samples  */
	  Ztrain[0] = iftExtractSamples(Ztmp,IFT_TRAIN);
	  Ztest [0] = iftExtractSamplesNotStatus(Ztmp,IFT_TRAIN);
	  iftSetStatus(Ztest[0],IFT_TEST);
	  iftDestroyDataSet(&Ztmp);

	  /* GridSearch */
	  float *valuesGrid;
	  valuesGrid = iftAllocFloatArray(2);
	  valuesGrid = iftGridSearch(Ztrain[0], percTrainVal, nRUNVal);
	  C = valuesGrid[0];
	  sigma = valuesGrid[1];
	  free(valuesGrid);

	  /* Train */
	  Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
	  iftDestroyDataSet(&Ztrain[0]);
	  Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
	  iftDestroyDataSet(&Ztest [0]);

	  svm = iftCreateRBFSVC(C, sigma);
	  iftSVMTrainOVO(svm,Ztrain[1]);

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

		  svm = iftCreateRBFSVC(C, sigma);
		  iftSVMTrainOVO(svm, Ztrain[1]);
		  iftSVMClassifyOVO(svm, Ztest[1], IFT_TEST); // Classification

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
		  values[iteration * ncols + 0] += iftCompTime(tic,toc) / (float)nrun;

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
		  values[iteration * ncols + 1] = (float)Z->ntrainsamples / (float)nrun;
		  printf("# train samples %d.\n", Z->ntrainsamples);

		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
		  values[iteration * ncols + 2] += nsamples / (float)nrun;
		  values[iteration * ncols + 3] += userCorrections / (float)nrun;

		  if (nsamples == 0) break;

		  nSamplesCorrected += userCorrections;
		  nSamplesShown += nsamples;

		  /* Select samples */
		  tic = iftTic();
		  Ztmp = iftCopyDataSet(Z, true);
		  Ztrain[0] = iftExtractSamples(Ztmp,IFT_TRAIN);
		  //Ztest [0] = iftExtractSamplesNotStatus(Ztmp,IFT_TRAIN);
		  Ztest [0] = iftCopyDataSet(Test, true);
		  iftSetStatus(Ztest[0],IFT_TEST);
		  iftDestroyDataSet(&Ztmp);

		  /* GridSearch */
		  if (iteration % 9 == 0){
			  float *valuesGrid;
			  valuesGrid = iftAllocFloatArray(2);
			  valuesGrid = iftGridSearch(Ztrain[0], percTrainVal, nRUNVal);
			  C = valuesGrid[0];
			  sigma = valuesGrid[1];
			  free(valuesGrid);
		  }

		  Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
		  iftDestroyDataSet(&Ztrain[0]);
		  Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
		  iftDestroyDataSet(&Ztest [0]);

		  svm = iftCreateRBFSVC(C, sigma);
		  iftSVMTrainOVO(svm,Ztrain[1]);
		  toc = iftToc();
		  values[iteration * ncols + 4] += iftCompTime(tic,toc) / (float)nrun;

		  /* Classify Z */
		  tic = iftTic();
		  iftSVMClassifyOVO(svm, Ztest[1], IFT_TEST); // Classification
		  accuracy_iter = iftTruePositives(Ztest[1]);
		  values[iteration * ncols + 5] += accuracy_iter  / (float)nrun;
		  toc = iftToc();
		  values[iteration * ncols + 6] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
		  tfinish = iftToc();
		  values[iteration * ncols + 7] += iftCompTime(tstart,tfinish) / (float)nrun;

		  iftDestroyDataSet(&Ztest[1]);
		  iftDestroySVM(svm);

		  iteration++;
	  } while (iteration < max_iteration);
	  printf("Finish AL SVM RBF\n");
	  iftDestroyDataSet(&reducedZ);
	  iftDestroyMST(&reducedMST);

	  iftDestroyKnnGraph(&graph);

	  return values;
}

float *iftRS_SVM_RBF(iftDataSet *Z, iftDataSet *Test, float kval, int max_iteration, int nrun){
	  timer *tstart, *tfinish;
	  timer *tic,*toc;
	  float accuracy_iter;
	  float *values;
	  int ncols = 8;
	  float C, sigma;
	  float percTrainVal = 0.33;
 	  int nRUNVal = 8;
	  values = iftAllocFloatArray(max_iteration*ncols);
	  C = 1e5;
	  sigma = 0.1;

      /* Get same amount of initial training samples */
	  //iftSelectRandTrainSamples(Z, Z->nlabels);
	  float trainPerc = (float) Z->nlabels / (float)Z->nsamples;
	  iftSelectSupTrainSamples(Z, trainPerc);

	  iftSVM      *svm   = NULL;
	  iftDataSet *Ztrain[2], *Ztest[2], *Ztmp;

	  int iteration = 0;
	  int nSamplesShown = 0;
	  int nSamplesCorrected = 0;
	  do {
		  tstart = iftTic();
		  printf("Iteration %d\n", iteration+1);

		  tic = iftTic();
		  /* Select samples for user labeling */
		  iftSet *selectedSamples = iftSelectRandSamplesForUserLabeling(Z, Z->nclasses * 2); // select 2 * number of classes
		  toc = iftToc();
		  values[iteration * ncols + 0] += iftCompTime(tic,toc)/ (float)nrun;

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
		  values[iteration * ncols + 1] = (float)Z->ntrainsamples / (float)nrun;
		  printf("# train samples %d.\n", Z->ntrainsamples);
		  printf("%d samples selected for user evaluation.\n", nsamples);
		  printf("%d user corrections.\n", userCorrections);
		  values[iteration * ncols + 2] += (float)nsamples/ (float)nrun;
		  values[iteration * ncols + 3] += (float)userCorrections/ (float)nrun;

		  if (nsamples == 0) break;

		  nSamplesCorrected += userCorrections;
		  nSamplesShown += nsamples;


		  /* Select samples */
		  tic = iftTic();
		  Ztmp = iftCopyDataSet(Z, true);
		  Ztrain[0] = iftExtractSamples(Ztmp,IFT_TRAIN);
		  //Ztest [0] = iftExtractSamplesNotStatus(Ztmp,IFT_TRAIN);
		  Ztest [0] = iftCopyDataSet(Test, true);
		  iftSetStatus(Ztest[0],IFT_TEST);
		  iftDestroyDataSet(&Ztmp);

		  /* GridSearch */
		  if (iteration % 9 == 0){
			  float *valuesGrid;
			  valuesGrid = iftAllocFloatArray(2);
			  valuesGrid = iftGridSearch(Ztrain[0], percTrainVal, nRUNVal);
			  C = valuesGrid[0];
			  sigma = valuesGrid[1];
			  free(valuesGrid);
		  }

		  /* Train */
		  Ztrain[1] = iftNormalizeDataSet(Ztrain[0]);
		  iftDestroyDataSet(&Ztrain[0]);
		  Ztest [1] = iftNormalizeTestDataSet(Ztrain[1],Ztest[0]);
		  iftDestroyDataSet(&Ztest [0]);

		  svm = iftCreateRBFSVC(C, sigma);
		  iftSVMTrainOVO(svm,Ztrain[1]);
		  toc = iftToc();
		  values[iteration * ncols + 4] = iftCompTime(tic,toc) / (float)nrun;

		  /* Classify Z */
		  tic = iftTic();
		  iftSVMClassifyOVO(svm, Ztest[1], IFT_TEST); // Classification
		  accuracy_iter = iftTruePositives(Ztest[1]);
		  values[iteration * ncols + 5] += accuracy_iter / (float)nrun;
		  toc = iftToc();
		  values[iteration * ncols + 6] += iftCompTime(tic,toc) / (float)nrun;
		  printf("Classification accuracy in the original dataset: %.2f%%\n", accuracy_iter * 100.0f);
		  tfinish = iftToc();
		  values[iteration * ncols + 7] += iftCompTime(tstart,tfinish) / (float)nrun;

		  printf("\n");

		  iftDestroyDataSet(&Ztrain[1]);
		  iftDestroyDataSet(&Ztest[1]);
		  iftDestroySVM(svm);

		  iteration++;

	  } while (iteration < max_iteration);
	  return values;
}


int main(int argc, char *argv[])
{

  iftDataSet      *dataset=NULL;
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

  if (argc != 9)
    iftError("Usage: iftActiveLearning <OPF dataset.dat> <kmax_perc [0,1]> <AL_OPF[csv]> <RS_OPF[csv]> <AL_SVM[csv]> <RS_SVM[csv]> <AL_SVM_RBF[csv]> <RS_SVM_RBF[csv]>","main");

  dataset = iftReadOPFDataSet(argv[1]);

  printf("Read dataset with %d samples.\n", dataset->nsamples);

  FILE *f1 = fopen(argv[3], "w");
  FILE *f2 = fopen(argv[4], "w");
  FILE *f3 = fopen(argv[5], "w");
  FILE *f4 = fopen(argv[6], "w");
  FILE *f5 = fopen(argv[7], "w");
  FILE *f6 = fopen(argv[8], "w");
  srand(time(NULL));



  float *time_selection, *time_train, *time_cls, *accuracy, *num_showed, *num_annotated, *time_iter, *ntrainsamples;
  int max_iteration = 25;
  int nmethods = 6;
  int nrun = 50;
  int ncols = 8;
  int init_ntrainsamples;
  time_selection = iftAllocFloatArray(max_iteration*nmethods);
  ntrainsamples = iftAllocFloatArray(max_iteration*nmethods);
  num_showed = iftAllocFloatArray(max_iteration*nmethods);
  num_annotated = iftAllocFloatArray(max_iteration*nmethods);
  time_train = iftAllocFloatArray(max_iteration*nmethods);
  accuracy = iftAllocFloatArray(max_iteration*nmethods);
  time_cls = iftAllocFloatArray(max_iteration*nmethods);
  time_iter = iftAllocFloatArray(max_iteration*nmethods);

  omp_set_num_threads(6);

  for (j = 0; j < nrun; j++) {

	  printf("nRun: %d\n", j);
  	  iftDataSet      *Z=NULL;
	  iftDataSet      *Test=NULL;
	  iftDataSet	  *ZTmp=NULL;
	  iftDataSet	  *TestTmp=NULL;
	  iftSelectSupTrainSamples(dataset, 0.80);
	  Z = iftExtractSamples(dataset,IFT_TRAIN);
	  Test = iftExtractSamples(dataset,IFT_TEST);

	  float k = atof(argv[2]);
	  float *values;


	  printf("********** AL OPF *************\n");
	  ZTmp = iftCopyDataSet(Z, true);
	  TestTmp = iftCopyDataSet(Test, true);
	  ZTmp->nlabels = Z->nlabels;
	  values = iftAL_OPF(ZTmp, TestTmp, k, max_iteration, nrun);
	  init_ntrainsamples = values[ncols*max_iteration];
	  printf("Init ntrainsamples: %d\n", init_ntrainsamples);
	  for(i = 0; i < max_iteration; i++){
		  time_selection[i*nmethods+0] += values[i*ncols + 0];
		  ntrainsamples[i*nmethods+0] += values[i*ncols + 1];
		  num_showed[i*nmethods+0] += values[i*ncols + 2];
		  num_annotated[i*nmethods+0] += values[i*ncols + 3];
		  time_train[i*nmethods+0] += values[i*ncols + 4];
		  accuracy[i*nmethods+0] += values[i*ncols + 5];
		  time_cls[i*nmethods+0] += values[i*ncols + 6];
		  time_iter[i*nmethods+0] += values[i*ncols + 7];
	  }
	  printf("after for AL OPF\n");
	  free(values);
	  //iftDestroyDataSet(&ZTmp);
	  iftDestroyDataSet(&TestTmp);

	  printf("********** RS OPF *************\n");
	  ZTmp = iftCopyDataSet(Z, true);
	  TestTmp = iftCopyDataSet(Test, true);
	  //ZTmp->nlabels = init_ntrainsamples;
	  ZTmp->nlabels = init_ntrainsamples;
	  values = iftRS_OPF(ZTmp, TestTmp, k, max_iteration, nrun);
	  for(i = 0; i < max_iteration; i++){
		  time_selection[i*nmethods+1] += values[i*ncols + 0];
		  ntrainsamples[i*nmethods+1] += values[i*ncols + 1];
		  num_showed[i*nmethods+1] += values[i*ncols + 2];
		  num_annotated[i*nmethods+1] += values[i*ncols + 3];
		  time_train[i*nmethods+1] += values[i*ncols + 4];
		  accuracy[i*nmethods+1] += values[i*ncols + 5];
		  time_cls[i*nmethods+1] += values[i*ncols + 6];
		  time_iter[i*nmethods+1] += values[i*ncols + 7];
	  }
	  printf("after for RS OPF\n");
	  free(values);
	  //iftDestroyDataSet(&ZTmp);
	  iftDestroyDataSet(&TestTmp);

	  printf("********** AL OPF RDS *************\n");
	  ZTmp = iftCopyDataSet(Z, true);
	  TestTmp = iftCopyDataSet(Test, true);
	  ZTmp->nlabels = Z->nlabels;
	  values = iftAL_OPF_RDS(ZTmp, TestTmp, k, max_iteration, nrun);
	  init_ntrainsamples = values[ncols*max_iteration];
	  printf("Init ntrainsamples: %d\n", init_ntrainsamples);
	  for(i = 0; i < max_iteration; i++){
		  time_selection[i*nmethods+2] += values[i*ncols + 0];
		  ntrainsamples[i*nmethods+2] += values[i*ncols + 1];
		  num_showed[i*nmethods+2] += values[i*ncols + 2];
		  num_annotated[i*nmethods+2] += values[i*ncols + 3];
		  time_train[i*nmethods+2] += values[i*ncols + 4];
		  accuracy[i*nmethods+2] += values[i*ncols + 5];
		  time_cls[i*nmethods+2] += values[i*ncols + 6];
		  time_iter[i*nmethods+2] += values[i*ncols + 7];
	  }
	  printf("after for AL OPF RDS\n");
	  free(values);
	  //iftDestroyDataSet(&ZTmp);
	  iftDestroyDataSet(&TestTmp);


	  printf("********** AL OPF MU *************\n");
	  ZTmp = iftCopyDataSet(Z, true);
	  TestTmp = iftCopyDataSet(Test, true);
	  //ZTmp->nlabels = Z->nlabels;
	  //ZTmp->nlabels = init_ntrainsamples;
	  ZTmp->nlabels = init_ntrainsamples;
	  values = iftAL_OPF_MU(ZTmp, TestTmp, k, max_iteration, nrun);
	  init_ntrainsamples = values[ncols*max_iteration];
	  printf("Init ntrainsamples: %d\n", init_ntrainsamples);
	  for(i = 0; i < max_iteration; i++){
		  time_selection[i*nmethods+3] += values[i*ncols + 0];
		  ntrainsamples[i*nmethods+3] += values[i*ncols + 1];
		  num_showed[i*nmethods+3] += values[i*ncols + 2];
		  num_annotated[i*nmethods+3] += values[i*ncols + 3];
		  time_train[i*nmethods+3] += values[i*ncols + 4];
		  accuracy[i*nmethods+3] += values[i*ncols + 5];
		  time_cls[i*nmethods+3] += values[i*ncols + 6];
		  time_iter[i*nmethods+3] += values[i*ncols + 7];
	  }
	  printf("after for AL OPF MU\n");
	  free(values);
	  //iftDestroyDataSet(&ZTmp);
	  iftDestroyDataSet(&TestTmp);

	  /*
	  printf("********** AL SVM *************\n");
	  ZTmp = iftCopyDataSet(Z, true);
	  TestTmp = iftCopyDataSet(Test, true);
	  ZTmp->nlabels = Z->nlabels;
	  values = iftAL_SVM(ZTmp, TestTmp, k, max_iteration, nrun);
	  init_ntrainsamples = values[ncols*max_iteration];
	  printf("Init ntrainsamples: %d\n", init_ntrainsamples);
	  for(i = 0; i < max_iteration; i++){
		  time_selection[i*nmethods+4] += values[i*ncols + 0];
		  ntrainsamples[i*nmethods+4] += values[i*ncols + 1];
		  num_showed[i*nmethods+4] += values[i*ncols + 2];
		  num_annotated[i*nmethods+4] += values[i*ncols + 3];
		  time_train[i*nmethods+4] += values[i*ncols + 4];
		  accuracy[i*nmethods+4] += values[i*ncols + 5];
		  time_cls[i*nmethods+4] += values[i*ncols + 6];
		  time_iter[i*nmethods+4] += values[i*ncols + 7];
	  }
	  printf("after for AL SVM\n");
	  free(values);
	  printf("after free values\n");
	  iftDestroyDataSet(&TestTmp);
	  printf("after free TestTmp\n");
	  //iftDestroyDataSet(&ZTmp);
	  printf("after free ZTmp\n");

	  printf("********** RS SVM *************\n");
	  ZTmp = iftCopyDataSet(Z, true);
	  TestTmp = iftCopyDataSet(Test, true);
	  ZTmp->nlabels = init_ntrainsamples;
	  values = iftRS_SVM(ZTmp, TestTmp, k, max_iteration, nrun);
	  for(i = 0; i < max_iteration; i++){
		  time_selection[i*nmethods+5] += values[i*ncols + 0];
		  ntrainsamples[i*nmethods+5] += values[i*ncols + 1];
		  num_showed[i*nmethods+5] += values[i*ncols + 2];
		  num_annotated[i*nmethods+5] += values[i*ncols + 3];
		  time_train[i*nmethods+5] += values[i*ncols + 4];
		  accuracy[i*nmethods+5] += values[i*ncols + 5];
		  time_cls[i*nmethods+5] += values[i*ncols + 6];
		  time_iter[i*nmethods+5] += values[i*ncols + 7];
	  }
	  free(values);
	  //iftDestroyDataSet(&ZTmp);
	  iftDestroyDataSet(&TestTmp);

	  printf("********** AL SVM RBF*************\n");
	  ZTmp = iftCopyDataSet(Z, true);
	  TestTmp = iftCopyDataSet(Test, true);
	  ZTmp->nlabels = Z->nlabels;
	  values = iftAL_SVM_RBF(ZTmp, TestTmp, graph, k, max_iteration, nrun);
	  for(i = 0; i < max_iteration; i++){
		  time_selection[i*nmethods+4] += values[i*ncols + 0];
		  ntrainsamples[i*nmethods+4] += values[i*ncols + 1];
		  num_showed[i*nmethods+4] += values[i*ncols + 2];
		  num_annotated[i*nmethods+4] += values[i*ncols + 3];
		  time_train[i*nmethods+4] += values[i*ncols + 4];
		  accuracy[i*nmethods+4] += values[i*ncols + 5];
		  time_cls[i*nmethods+4] += values[i*ncols + 6];
		  time_iter[i*nmethods+4] += values[i*ncols + 7];
	  }
	  free(values);
	  iftDestroyDataSet(&TestTmp);
	  //iftDestroyDataSet(&ZTmp);

	  printf("********** RS SVM RBF *************\n");
	  ZTmp = iftCopyDataSet(Z, true);
	  TestTmp = iftCopyDataSet(Test, true);
	  ZTmp->nlabels = Z->nlabels;
	  values = iftRS_SVM_RBF(ZTmp, TestTmp, k, max_iteration, nrun);
	  for(i = 0; i < max_iteration; i++){
		  time_selection[i*nmethods+5] += values[i*ncols + 0];
		  ntrainsamples[i*nmethods+5] += values[i*ncols + 1];
		  num_showed[i*nmethods+5] += values[i*ncols + 2];
		  num_annotated[i*nmethods+5] += values[i*ncols + 3];
		  time_train[i*nmethods+5] += values[i*ncols + 4];
		  accuracy[i*nmethods+5] += values[i*ncols + 5];
		  time_cls[i*nmethods+5] += values[i*ncols + 6];
		  time_iter[i*nmethods+5] += values[i*ncols + 7];
	  }
	  free(values);
	  //iftDestroyDataSet(&ZTmp);
	  iftDestroyDataSet(&TestTmp);
      */

	  printf("before iftDestroyKnnGraph\n");
	  //iftDestroyKnnGraph(&graph);
	  printf("after iftDestroyKnnGraph\n");
	  //iftDestroyDataSet(&Test);
	  printf("after iftDestroyDataSet Test\n");
	  //iftDestroyDataSet(&Z);
  }

  for (j = 0; j < max_iteration; j++) {
  	fprintf(f1,"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", time_selection[j*nmethods+0], time_train[j*nmethods+0], time_cls[j*nmethods+0], accuracy[j*nmethods+0], num_showed[j*nmethods+0], num_annotated[j*nmethods+0], time_iter[j*nmethods+0], ntrainsamples[j*nmethods+0]);
  }

  for (j = 0; j < max_iteration; j++) {
  	fprintf(f2,"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", time_selection[j*nmethods+1], time_train[j*nmethods+1], time_cls[j*nmethods+1], accuracy[j*nmethods+1], num_showed[j*nmethods+1], num_annotated[j*nmethods+1], time_iter[j*nmethods+1], ntrainsamples[j*nmethods+1]);
  }

  for (j = 0; j < max_iteration; j++) {
  	fprintf(f3,"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", time_selection[j*nmethods+2], time_train[j*nmethods+2], time_cls[j*nmethods+2], accuracy[j*nmethods+2], num_showed[j*nmethods+2], num_annotated[j*nmethods+2], time_iter[j*nmethods+2], ntrainsamples[j*nmethods+2]);
  }

  for (j = 0; j < max_iteration; j++) {
    fprintf(f4,"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", time_selection[j*nmethods+3], time_train[j*nmethods+3], time_cls[j*nmethods+3], accuracy[j*nmethods+3], num_showed[j*nmethods+3], num_annotated[j*nmethods+3], time_iter[j*nmethods+3], ntrainsamples[j*nmethods+3]);
  }

  for (j = 0; j < max_iteration; j++) {
    fprintf(f5,"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", time_selection[j*nmethods+4], time_train[j*nmethods+4], time_cls[j*nmethods+4], accuracy[j*nmethods+4], num_showed[j*nmethods+4], num_annotated[j*nmethods+4], time_iter[j*nmethods+4], ntrainsamples[j*nmethods+4]);
  }

  for (j = 0; j < max_iteration; j++) {
    fprintf(f6,"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", time_selection[j*nmethods+5], time_train[j*nmethods+5], time_cls[j*nmethods+5], accuracy[j*nmethods+5], num_showed[j*nmethods+5], num_annotated[j*nmethods+5], time_iter[j*nmethods+5], ntrainsamples[j*nmethods+5]);
  }

  fclose(f1);
  fclose(f2);
  fclose(f3);
  fclose(f4);
  fclose(f5);
  fclose(f6);

  printf("\n");
  printf("Finish exp. Active Learning Test\n");

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
