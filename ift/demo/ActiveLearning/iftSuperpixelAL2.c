#include "ift.h"


//Filters training samples that are separable in the feature space.
//If two training_samples have the same group but different classes, the one in the background is eliminated.
iftSet* iftFilterSeparableTrainingSamples(iftDataSet *dataset, iftSet *training_samples, int *cluster_labels){
	iftSet *filtered = 0;

	iftSet *s1 = training_samples;
	while(s1){
		int s1_class = dataset->sample[s1->elem].truelabel;

		if(s1_class == 1){
			iftSet *s2 = training_samples;

			int s1_label = cluster_labels[s1->elem];

			int insert_s1 = 1;
			while(s2){
				int s2_label = cluster_labels[s2->elem];
				int s2_class = dataset->sample[s2->elem].truelabel;

				if( (s1_label == s2_label) && (s2_class != s1_class) ){
					insert_s1 = 0;
					break;
				}

				s2 = s2->next;
			}

			if(insert_s1)
				iftUnionSetElem(&filtered,s1->elem);
		}else{
			iftUnionSetElem(&filtered,s1->elem);
		}

		s1 = s1->next;
	}

	return filtered;

}

void draw_markers_on_superpixels(iftImage* img, iftDataSet* dataset, iftLabeledSet* centers, iftSet* superpixels, char* output_path, int use_label){
	iftImage *output = iftCopyImage(img);
	iftAdjRel *circle = iftCircular(4.0);

	iftColor yellow; yellow.val[0] = 76; yellow.val[1] = 85; yellow.val[2] = 255;
	iftColor red; red.val[0] = 225; red.val[1] = 1; red.val[2] = 148;

	iftLabeledSet *c = centers;
	while(c){
		int center = c->elem;
		int superpixel = c->label;

		if(iftSetHasElement(superpixels,superpixel - 1) ){
			int i;
			for(i = 0; i < circle->n; i++){
				iftVoxel u = iftGetVoxelCoord(img,center);
				iftVoxel v = iftGetAdjacentVoxel(circle,u,i);

				if(iftValidVoxel(img,v)){
					int p = iftGetVoxelIndex(img,v);

					int class;
					if(use_label)
						class = dataset->sample[superpixel - 1].label;
					else
						class = dataset->sample[superpixel - 1].truelabel;

					iftColor *color;
					if(class == 1)
						color = &yellow;
					else
						color = &red;

					output->val[p] = color->val[0];
					output->Cb[p] =  color->val[1];
					output->Cr[p] =  color->val[2];
				}
			}
		}

		c = c->next;
	}

	iftWriteImageP6(output,output_path);
}

int main(int argc, char **argv) {
	if(argc != 3 && argc != 6)
		printf("Usage: iftSuperpixelAL2 [image] [ground_truth] [spatial_radius] [volume_threshold] [kmax_perc]");

	//Creating dataset from image
	iftImage *img = iftReadImageP6(argv[1]);
	iftImage *gt = iftReadImageP5(argv[2]);

	float kmax_perc = 0.10;
	float spatial_radius = 1.5;
	float volume_thershold = 1500;
	if(argc == 6){
		spatial_radius = atof(argv[3]);
		volume_thershold = atof(argv[4]);
		kmax_perc = atof(argv[5]);
	}

	iftAdjRel *A = iftCircular(spatial_radius);
	iftImage *basins = iftImageBasins(img,A);
	iftImage* marker = iftVolumeClose(basins, volume_thershold);

	iftImage* label_image = iftWaterGray(basins,marker,A);

	iftDataSet *dataset = iftSupervoxelsToDataSet(img,label_image);
	if(dataset->nfeats == 3)
		dataset->alpha[0] = 0.2; dataset->alpha[1] = 1.0; dataset->alpha[2] = 1.0;

	//Setting classes
	iftSetSuperpixelClassesFromGroundTruth(label_image,gt,dataset);

	iftLabeledSet *centers = iftGeodesicCenters(label_image);

	//Draws all markers on superpixels (for testing)
	iftSet *sp_set = 0;
	int n_superpixels = iftMaximumValue(label_image);
	int i;
	for(i = 0; i < n_superpixels; i++)
		iftUnionSetElem(&sp_set, i);
	draw_markers_on_superpixels(img,dataset,centers,sp_set,"all_markers.ppm",0);

	//Creates the clustering graph
	iftKnnGraph *knn_graph = iftUnsupLearn(dataset, 1.0, kmax_perc, iftNormalizedCut, 10);
	iftUnsupClassify(knn_graph, dataset);

	iftImage* clustering_image = iftSuperpixelLabelImageFromDataset(label_image,dataset);
	iftImage* normalized_cl_image = iftNormalize(clustering_image,0,255);
	iftWriteImageP5(normalized_cl_image,"clusters.ppm");
	iftDestroyImage(&clustering_image);
	iftDestroyImage(&normalized_cl_image);

	//Stores the labels for each sample
	int *cluster_labels = iftAllocIntArray(dataset->nsamples);
	for(i = 0; i < dataset->nsamples; i++)
		cluster_labels[i] = dataset->sample[i].label;

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
	draw_markers_on_superpixels(img,dataset,centers,rootSet,"all_roots.pgm",0);

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

		iftSet *unfiltered_samples = iftSetConcat(trainSamples,newSamples);

		char buffer[31];
		sprintf(buffer,"markers_iter_%d",i + 1);
		draw_markers_on_superpixels(img,dataset,centers,unfiltered_samples,buffer, 1);

		iftSet *oldSamples = trainSamples;
		trainSamples = iftFilterSeparableTrainingSamples(dataset,unfiltered_samples,cluster_labels);

		iftDestroySet(&newSamples);
		iftDestroySet(&unfiltered_samples);
		iftDestroySet(&oldSamples);

		iftDestroyCplGraph(&cl_graph);
	}

}
