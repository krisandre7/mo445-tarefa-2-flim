#include "ift.h"

//Toggles debugging information
#ifndef IFTACTIVELEARNING2_DEBUG
	#define IFTACTIVELEARNING2_DEBUG
#endif

int main(int argc, char **argv) {
	if (argc != 4)
		iftError("Usage: iftActiveLearning2 <dataset.dat> <train_perc [0,1]> <kmax_perc [0,1]>","main");

	//Reads the dataset
	iftDataSet *dataset = iftReadOPFDataSet(argv[1]);

	//Creates the clustering graph
	iftKnnGraph *knn_graph = iftUnsupLearn(dataset, atof(argv[2]), atof(argv[3]), iftNormalizedCut, 10);
	iftUnsupClassify(knn_graph, dataset);

	//Obtains root set and boundary set
	iftSet *rootSet = iftGetKnnRootSamples(knn_graph);
	iftSet  *boundarySet = iftGetKnnBoundarySamples(knn_graph);

	iftDestroyKnnGraph(&knn_graph);

	//Obtains the MST from the complete graph defined by the boundary set
	iftMST *mst = iftCreateMSTFromSet(dataset, boundarySet);
	//iftNormalizeSampleWeightMST(mst);
	iftSortNodesByWeightMST(mst, DECREASING);

	iftDestroySet(&boundarySet);

	//The initial training set composed by the labeled roots
	iftSet  *trainSamples = rootSet;
	#ifdef IFTACTIVELEARNING2_DEBUG
	printf(" Roots: \n");
	iftSet *s = trainSamples;
	while(s){
		printf("%d,",s->elem);
		s = s->next;
	}
	printf("\n");
	#endif

	int i;
	int niters = 10;
	int samples_iter = 10;
	for(i = 0; i < niters; i++){
		//Creates a new classifier assuming that the correct classes have been assigned to the trainSamples
		iftSetStatus(dataset,IFT_TEST);
		iftSetStatusForSet(dataset, trainSamples, IFT_TRAIN);
		iftCplGraph *cl_graph = iftCreateCplGraph(dataset);
		iftSupTrain(cl_graph);

		//Selects new samples from the MST and marks those already selected
		iftSet *newSamples = iftGetMSTBoundarySamples(cl_graph, mst, samples_iter);
		#ifdef IFTACTIVELEARNING2_DEBUG
		printf("MST nodes (iteration %d): \n", i + 1);
		s = newSamples;
		while(s){
			printf("%d,",s->elem);
			s = s->next;
		}
		printf("\n");
		#endif

		iftDestroyCplGraph(&cl_graph);

		iftSet *oldSamples = trainSamples;
		trainSamples = iftSetConcat(trainSamples,newSamples);

		iftDestroySet(&newSamples);
		iftDestroySet(&oldSamples);
	}

	iftDestroyMST(&mst);
	iftDestroySet(&trainSamples);
	iftDestroyDataSet(&dataset);

}
