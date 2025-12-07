#include "ift.h"

void fix_slash(char *directory_path){
	if(directory_path[strlen(directory_path) -1 ] != '/' )
		strcat(directory_path, "/");
}

void create_dir(char **argv){
	char buffer[256];

	DIR *dir = opendir(argv[1]);
	if (dir == NULL)
		iftError("Could not open input directory","ift_lesion_seg");
	else
		closedir(dir);

	dir = opendir(argv[2]);
	if (dir == NULL)
		mkdir(argv[2],0700);
	else
		closedir(dir);

	strcpy(buffer, argv[2]);
	fix_slash(buffer);
	strcat(buffer, "vis/");
	dir = opendir(buffer);
	if (dir == NULL)
		mkdir(buffer,0700);
	else
		closedir(dir);

	strcpy(buffer, argv[2]);
	fix_slash(buffer);
	strcat(buffer, "labels/");
	dir = opendir(buffer);
	if (dir == NULL)
		mkdir(buffer,0700);
	else
		closedir(dir);
}

int main(int argc, char **argv) {
	if(argc < 4)
			iftError("Usage: lesion_superseg DATASET OUTPUT NREGIONS", "lesion_superseg");

	//Superpixel parameters
	int volume_threshold = 150;
	float spatial_radius = 3.0;
	
	iftImage *tmp;

	char input_dir[256];
	strcpy(input_dir, argv[1]);
	fix_slash(input_dir);
	strcat(input_dir, "orig/");

	char input_label_dir[256];
	strcpy(input_label_dir, argv[1]);
	fix_slash(input_label_dir);
	strcat(input_label_dir, "labels/");

	int nimages = iftCountImageNames(input_dir, "ppm");
	iftImageNames *img_names = iftCreateAndLoadImageNames(nimages, input_dir, "ppm");

	create_dir(argv);

	iftColor blue;
	blue.val[0] = 255; blue.val[1] = 255; blue.val[2] = 0;
	blue = iftRGBtoYCbCr(blue);

	char output_vis[256];
	strcpy(output_vis, argv[2]);
	fix_slash(output_vis);
	strcat(output_vis, "vis/");

	char output_label[256];
	strcpy(output_label, argv[2]);
	fix_slash(output_label);
	strcat(output_label, "labels/");

	iftAdjRel *adj = iftCircular(spatial_radius);
	iftAdjRel *adj1 = iftCircular(1.5);
	iftAdjRel *adj0 = iftCircular(0.0);

	char name_buffer[256];
	for(int i = 0; i < nimages; i++){
		strcpy(name_buffer, input_dir);
		strcat(name_buffer, img_names[i].image_name);

		printf("Processing image %s.\n",name_buffer);
		iftImage *image = iftReadImageP6(name_buffer);

		char *base_name = iftSplitStringAt(img_names[i].image_name, ".ppm",0);

		strcpy(name_buffer, input_label_dir);
		strcat(name_buffer, base_name);
		strcat(name_buffer, ".pgm");

		iftImage *segmented = iftReadImageP5(name_buffer);

		iftImage *basins = iftImageBasins(image, adj);
		iftImage *marker = iftVolumeClose(basins, volume_threshold);
		iftImage *label_image = iftWaterGray(basins, marker, adj);
		
		//Masks the image using the segmentation
		tmp = label_image;
		label_image = iftMask(label_image, segmented);
		iftDestroyImage(&tmp);

		int p;

		//Relabels from 0 to n, where n is the number of regions inside the segmentation
		label_image = iftRelabelRegions(label_image, adj1);

		//Relabels from 1 to n+1, to include the background region
		for(p = 0; p < label_image->n; p++)
			label_image->val[p]++;

		iftDataSet *dataset = iftSupervoxelsToMeanSizeDataSet(image, label_image, YCbCr_CSPACE);
		//The first region (background) should not have a valid feature
		for(p = 0; p < dataset->nfeats; p++)
			dataset->sample[0].feat[p] = -1;
		dataset->alpha[0] = 0.2;
		dataset->alpha[3] = 0.0;

		iftRegionHierarchy *rh = iftCreateRegionHierarchy(dataset, adj1, iftMergeMeanSizeSupervoxel);
		iftImage *merged_label = iftFlattenRegionHierarchy(rh, atoi(argv[3]));
		iftImage *drawn_labels = iftCopyImage(image);

		iftDrawBorders(drawn_labels,merged_label,adj1,blue,adj0);

		char output_buffer[256];
		strcpy(output_buffer, output_vis);
		strcat(output_buffer, img_names[i].image_name);
		iftWriteImageP6(drawn_labels, output_buffer);

		strcpy(output_buffer, output_label);
		strcat(output_buffer, base_name);
		strcat(output_buffer, ".pgm");
		iftWriteImageP5(merged_label, output_buffer);

		free(base_name);
		iftDestroyImage(&drawn_labels);
		iftDestroyImage(&image);
		iftDestroyImage(&marker);
		iftDestroyImage(&segmented);
		iftDestroyImage(&label_image);
		iftDestroyImage(&merged_label);
	}

	iftDestroyAdjRel(&adj);
	iftDestroyAdjRel(&adj1);
	iftDestroyAdjRel(&adj0);
	iftDestroyImageNames(img_names);

	return 0;
}
