#include "ift.h"

void fix_slash(char *directory_path){
	if(directory_path[strlen(directory_path) -1 ] != '/' )
		strcat(directory_path, "/");
}

int main(int argc, char **argv) {
	if(argc < 2)
		iftError("Usage: create_dataset INPUT","create_dataset");

	int BINS_PER_BAND_HIST = 16;
	int BINS_PER_BAND_BIC = 4;
	int SHOG_BINS = 16;

	char input_path[256];
	strcpy(input_path, argv[1]);
	fix_slash(input_path);

	DIR *dir = opendir(input_path);
	if (dir == NULL)
		iftError("Could not open input directory","create_dataset");
	else
		closedir(dir);

	//Images
	char orig_path[256];
	strcpy(orig_path, input_path);
	strcat(orig_path, "orig/");

	//Oversegmentations
	char overseg_path[256];
	strcpy(overseg_path, orig_path);
	strcat(overseg_path, "overseg/");

	//Markers
	char markers_path[256];
	strcpy(markers_path, orig_path);
	strcat(markers_path, "markers/");

	//Output path
	char output_path[256];
	strcpy(output_path, orig_path);
	strcat(output_path, "datasets/");

	dir = opendir(output_path);
	if (dir == NULL)
		mkdir(output_path,0700);
	else
		closedir(dir);

	//The oversegmentations folder is used to determine which images have been annotated
	int nimages = iftCountImageNames(overseg_path, "pgm");
	iftImageNames *img_names = iftCreateAndLoadImageNames(nimages, overseg_path, "pgm");

	char name_buffer[256];
	for(int i = 0; i < nimages; i++){
		strcpy(name_buffer, overseg_path);
		strcat(name_buffer, img_names[i].image_name);

		printf("Processing image %s.\n",name_buffer);

		iftImage *label = iftReadImageP5(name_buffer);

		char *base_name = iftSplitStringAt(img_names[i].image_name, ".pgm",0);

		strcpy(name_buffer, orig_path);
		strcat(name_buffer, base_name);
		strcat(name_buffer, ".ppm");

		iftImage *image = iftReadImageP6(name_buffer);

		//Five different descriptors
		iftDataSet **datasets = (iftDataSet**)calloc(5,sizeof(iftDataSet*));

		datasets[0] = iftSupervoxelsToLabColorMomentsDataset(image, label);
		datasets[1] = iftSupervoxelsToLabHistogramDataset(image, label, BINS_PER_BAND_HIST);
		datasets[2] = iftSupervoxelsToBICDataset(image, label, BINS_PER_BAND_BIC);
		datasets[3] = iftSupervoxelsToUniformLBP2D(image, label);
		datasets[4] = iftSupervoxelsToSimplifiedHOG2D(image, label, SHOG_BINS);

		iftDataSet *dataset = iftConcatDatasetFeatures(datasets, 5);

		//Store classes annotated by user
		strcpy(name_buffer, markers_path);
		strcat(name_buffer, base_name);
		strcat(name_buffer, ".txt");

		iftLabeledSet *markers = iftReadSeeds2D(name_buffer, image);
		while(markers){
			int r = label->val[markers->elem] - 1;
			if(r >= 0){
				dataset->sample[r].truelabel = markers->label;
			}

			markers = markers->next;
		}

		//Write this dataset to disc
		strcpy(name_buffer, output_path);
		strcat(name_buffer, base_name);
		strcat(name_buffer, ".opf");
		iftWriteOPFDataSet(dataset,name_buffer);

		free(base_name);
		iftDestroyImage(&image);
		iftDestroyImage(&label);

		int i;
		for(i = 0; i < 5; i++)
			iftDestroyDataSet(&datasets[i]);

		int r;
		for(r = 0; r < dataset->nsamples; r++){
			for(i = 0; i < dataset->nfeats; i++)
				printf("%f ",dataset->sample[r].feat[i]);
			printf("\n");
		}

		iftDestroyDataSet(&dataset);
	}

	iftDestroyImageNames(img_names);

	return 0;
}
