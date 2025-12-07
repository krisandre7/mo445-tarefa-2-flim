#include "ift.h"

iftImage *ift_segmentation_superpixel(iftImage *image, iftLabeledSet *seeds, float spatial_radius, int volume_threshold){
	iftAdjRel *adj = iftCircular(spatial_radius);
	iftAdjRel *adj1 = iftCircular(1.0);

	iftImage *basins = iftImageBasins(image, adj);
//	iftImage *basins = iftExpImageBasins(image, adj);
	iftImage *marker = iftVolumeClose(basins, volume_threshold);
	iftImage *label = iftWaterGray(basins, marker, adj);

	iftDataSet *dataset = iftSupervoxelsToDataSet(image, label);
	dataset->alpha[0] = 0.2;

	iftRegionGraph *region_graph = iftRegionGraphFromLabelImage(label, dataset, adj1);
	iftSuperpixelClassification(region_graph, label, seeds);

	iftImage *result = iftSuperpixelLabelImageFromDataset(label, dataset);

	iftDestroyAdjRel(&adj);
	iftDestroyAdjRel(&adj1);
	iftDestroyImage(&basins);
	iftDestroyImage(&marker);
	iftDestroyImage(&label);
	iftDestroyDataSet(&dataset);
	iftDestroyRegionGraph(&region_graph);

	return result;
}

iftImage *ift_segmentation_pixel(iftImage *image, iftLabeledSet *seeds){
	iftAdjRel *adj = iftCircular(1.0);

	iftImage *basins = iftImageBasins(image, adj);
//	iftImage *basins = iftExpImageBasins(image, adj);
	iftImage *label = iftWatershed(basins,adj, seeds);

	iftDestroyAdjRel(&adj);
	iftDestroyImage(&basins);

	return label;
}

iftLabeledSet* seeds_from_preliminary_seg(iftImage *label_image, float dist_threshold){
	iftAdjRel *adj = iftCircular(1.5);
	iftImage *dist = iftEuclDistTrans(label_image, adj, BOTH, NULL, NULL, NULL);

	iftLabeledSet *seed = 0;
	int p;
	for(p = 0; p < label_image->n; p++){
		if(sqrt(dist->val[p]) >= dist_threshold){
			iftInsertLabeledSet(&seed,p, label_image->val[p]);
		}
	}

	iftDestroyAdjRel(&adj);
	iftDestroyImage(&dist);

	return seed;
}

void fix_slash(char *directory_path){
	if(directory_path[strlen(directory_path) -1 ] != '/' )
		strcat(directory_path, "/");
}

void create_dirs(char **argv){
	char buffer[256];

	DIR *dir = opendir(argv[1]);
	if (dir == NULL)
		iftError("Could not open input directory","ift_lesion_seg");
	else
		closedir(dir);

	dir = opendir(argv[2]);
	if (dir == NULL)
		iftError("Could not open preliminary segmentation directory","ift_lesion_seg");
	else
		closedir(dir);

	dir = opendir(argv[3]);
	if (dir == NULL)
		mkdir(argv[3],0700);
	else
		closedir(dir);

	strcpy(buffer, argv[3]);
	fix_slash(buffer);
	strcat(buffer, "vis/");
	dir = opendir(buffer);
	if (dir == NULL)
		mkdir(buffer,0700);
	else
		closedir(dir);

	strcpy(buffer, argv[3]);
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
		iftError("Usage: ift_lesion_seg DATASET SEGMENTED_DATASET OUTPUT [DIST_THRESHOLD]", "ift_lesion_seg");

	//Superpixel parameters:
	int volume_threshold = 150;
	float spatial_radius = 1.5;

	char input_dir[256];
	strcpy(input_dir, argv[1]);
	fix_slash(input_dir);
	strcat(input_dir, "orig/");

	int nimages = iftCountImageNames(input_dir, "ppm");
	iftImageNames *img_names = iftCreateAndLoadImageNames(nimages, input_dir, "ppm");

	char prelim_seg_dir[256];
	strcpy(prelim_seg_dir, argv[2]);
	fix_slash(prelim_seg_dir);
	strcat(prelim_seg_dir, "labels/");

	float dist_threshold = 10;
	if(argc > 4)
		dist_threshold = atof(argv[4]);

	create_dirs(argv);

	iftColor blue;
	blue.val[0] = 0; blue.val[1] = 0; blue.val[2] = 255;
	blue = iftRGBtoYCbCr(blue);

	char output_vis[256];
	strcpy(output_vis, argv[3]);
	fix_slash(output_vis);
	strcat(output_vis, "vis/");

	char output_label[256];
	strcpy(output_label, argv[3]);
	fix_slash(output_label);
	strcat(output_label, "labels/");

	iftAdjRel *adj = iftCircular(1.5);
	iftAdjRel *adj0 = iftCircular(0.0);

	char name_buffer[256];
	for(int i = 0; i < nimages; i++){
		strcpy(name_buffer, input_dir);
		strcat(name_buffer, img_names[i].image_name);

		printf("Processing image %s.\n",name_buffer);
		iftImage *image = iftReadImageP6(name_buffer);

		char *base_name = iftSplitStringAt(img_names[i].image_name, ".ppm",0);

		strcpy(name_buffer, prelim_seg_dir);
		strcat(name_buffer, base_name);
		strcat(name_buffer, ".pgm");
		iftImage *prelim_label = iftReadImageP5(name_buffer);

		iftLabeledSet* seeds = seeds_from_preliminary_seg(prelim_label, dist_threshold);

		iftImage *label = ift_segmentation_superpixel(image, seeds, spatial_radius, volume_threshold);
		//iftImage *label = ift_segmentation_pixel(image, seeds);

		iftImage *drawn_labels = iftCopyImage(image);
		iftDrawBorders(drawn_labels,label,adj,blue,adj0);

		char output_buffer[256];
		strcpy(output_buffer, output_vis);
		strcat(output_buffer, img_names[i].image_name);
		iftWriteImageP6(drawn_labels, output_buffer);

		strcpy(output_buffer, output_label);
		strcat(output_buffer, base_name);
		strcat(output_buffer, ".pgm");
		iftWriteImageP5(label, output_buffer);

		free(base_name);
		iftDestroyImage(&drawn_labels);
		iftDestroyImage(&image);
		iftDestroyImage(&label);
		iftDestroyLabeledSet(&seeds);
	}

	iftDestroyAdjRel(&adj);
	iftDestroyAdjRel(&adj0);
	iftDestroyImageNames(img_names);

	return 0;
}
