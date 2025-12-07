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

	iftColor red; red.val[0] = 76; red.val[1] = 85; red.val[2] = 255;
	iftColor yellow; yellow.val[0] = 225; yellow.val[1] = 1; yellow.val[2] = 148;

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
						color = &red;
					else
						color = &yellow;

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

void write_seeds(char *filename, iftDataSet *dataset, iftSet *delineationSeeds, iftLabeledSet *centers, iftImage *img){
	iftLabeledSet *ls = 0;

	//draw around adjacency
	iftAdjRel *circle = iftCircular(5.0);

	iftLabeledSet *c = centers;
	while(c){
		int center = c->elem;
		int superpixel = c->label;

		if(iftSetHasElement(delineationSeeds,superpixel - 1) ){
			int i;
			for(i = 0; i < circle->n; i++){
				iftVoxel u = iftGetVoxelCoord(img,center);
				iftVoxel v = iftGetAdjacentVoxel(circle,u,i);

				if(iftValidVoxel(img,v)){
					int p = iftGetVoxelIndex(img,v);

					iftInsertLabeledSet(&ls,p,dataset->sample[superpixel - 1].truelabel-1);
				}
			}
		}

		c = c->next;
	}

	iftWriteSeeds2D(filename,ls,img);

	iftDestroyLabeledSet(&ls);
	iftDestroyAdjRel(&circle);

}

int main(int argc, char **argv) {
	if(argc != 3 && argc != 6)
		iftError("Usage: iftSuperpixelConn [image] [ground_truth] [spatial_radius] [volume_threshold] [kmax_perc]","main");

	//Creating dataset from image
	iftImage *img = iftReadImageP6(argv[1]);
	iftImage *gt = iftReadImageP5(argv[2]);

	float kmax_perc = 0.10;
	float spatial_radius = 1.5;
	float volume_threshold = 1000;
	if(argc == 6){
		spatial_radius = atof(argv[3]);
		volume_threshold = atof(argv[4]);
		kmax_perc = atof(argv[5]);
	}

	iftAdjRel *A = iftCircular(spatial_radius);
	iftImage *basins = iftImageBasins(img,A);
	iftImage* marker = iftVolumeClose(basins, volume_threshold);

	iftImage* label_image = iftWaterGray(basins,marker,A);

	iftDataSet *dataset = iftSupervoxelsToDataSet(img,label_image);
	if(dataset->nfeats == 3)
		dataset->alpha[0] = 0.2; dataset->alpha[1] = 1.0; dataset->alpha[2] = 1.0;

	//Setting classes
	iftSetSuperpixelClassesFromGroundTruth(label_image,gt,dataset);

	iftLabeledSet *centers = iftGeodesicCenters(label_image);

	iftImage *img_superpixels = iftCopyImage(img);
	iftAdjRel *four_neig = iftCircular(1.0);
	iftAdjRel *zero_neig = iftCircular(0.0);
	iftColor white; white.val[0] = 255; white.val[1] = 128; white.val[2] = 128;
	iftDrawBorders(img_superpixels,label_image,four_neig,white,zero_neig);

	//Paint image based on the ground truth for testing
	iftColor red; red.val[0] = 76; red.val[1] = 85; red.val[2] = 255;
//	iftColor yellow; yellow.val[0] = 225; yellow.val[1] = 1; yellow.val[2] = 148;
	iftImage *img_all_markers = iftCopyImage(img_superpixels);
	int p;
	for(p = 0; p < img_all_markers->n; p++){
		if(dataset->sample[label_image->val[p] - 1].truelabel == 1){
			img_all_markers->Cb[p] = red.val[1];
			img_all_markers->Cr[p] = red.val[2];
		}
	}
	iftWriteImageP6(img_all_markers,"ground_truth.ppm");
	iftDestroyImage(&img_all_markers);

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
	int i;
	for(i = 0; i < dataset->nsamples; i++)
		cluster_labels[i] = dataset->sample[i].label;

	iftRegionGraph *region_graph = iftRegionGraphFromLabelImage(label_image,dataset,four_neig);

	//Obtains root set
	iftSet *rootSet = iftGetKnnRootSamples(knn_graph);

	iftDestroyKnnGraph(&knn_graph);

	iftMST *mst = iftCreateSuperpixelActiveLearningMST(region_graph);
	iftSortNodesByWeightMST(mst, DECREASING);

	//The initial training set composed by the labeled roots
	iftSet  *trainSamples = rootSet;
	draw_markers_on_superpixels(img_superpixels,dataset,centers,rootSet,"roots.ppm",0);
	iftSetNodeColors(mst,trainSamples,BLACK);

	//show delineation markers and segmentation for each iteration

	iftSet *delineationSeeds = iftSetCopy(trainSamples);

	int niters = 20;
	int samples_iter = 4;
	for(i = 0; i < niters; i++){
		//Creates a new classifier assuming that the correct classes have been assigned to the trainSamples
		iftSetStatus(dataset,IFT_TEST);
		iftSetStatusForSet(dataset, trainSamples, IFT_TRAIN);
		iftCplGraph *cl_graph = iftCreateCplGraph(dataset);
		iftSupTrain(cl_graph);

		//Selects new samples from the MST and marks those already selected
		iftSet *newSamples = iftGetMSTBoundarySamples(cl_graph, mst, samples_iter);

		//Stores the seeds for delineation
		iftSet* old_delineation = delineationSeeds;
		delineationSeeds = iftSetConcat(delineationSeeds,newSamples);
		iftDestroySet(&old_delineation);

		iftSet *unfiltered_samples = iftSetConcat(trainSamples,newSamples);

		iftSet *oldSamples = trainSamples;
		trainSamples = iftFilterSeparableTrainingSamples(dataset,unfiltered_samples,cluster_labels);
		iftDestroySet(&oldSamples);

		char buffer[31];
		sprintf(buffer,"trainining_%d.ppm",i + 1);
		draw_markers_on_superpixels(img_superpixels,dataset,centers,trainSamples,buffer, 0);

		buffer[0] = 0;
		sprintf(buffer,"delineation_%d.ppm",i + 1);
		draw_markers_on_superpixels(img_superpixels,dataset,centers,delineationSeeds,buffer, 0);
		buffer[0] = 0;
		sprintf(buffer,"seeds_%d.txt",i + 1);
		write_seeds(buffer,dataset,delineationSeeds, centers, img);

		buffer[0] = 0;
		sprintf(buffer,"suggestions_%d.ppm",i + 1);
		draw_markers_on_superpixels(img_superpixels,dataset,centers,newSamples,buffer, 1);

		iftDestroySet(&newSamples);
		iftDestroySet(&unfiltered_samples);
		iftDestroyCplGraph(&cl_graph);
	}

	iftDestroyImage(&img);
	iftDestroyImage(&gt);
	iftDestroyAdjRel(&A);
	iftDestroyImage(&basins);
	iftDestroyImage(&marker);

	iftDestroyImage(&label_image);

	iftDestroyDataSet(&dataset);
	iftDestroyLabeledSet(&centers);

	iftDestroyImage(&img_superpixels);

	iftDestroyAdjRel(&four_neig);
	iftDestroyAdjRel(&zero_neig);

	free(cluster_labels);

	iftDestroyRegionGraph(&region_graph);

	iftDestroySet(&trainSamples);
	iftDestroySet(&delineationSeeds);

	iftDestroyMST(&mst);
}
