#include "ift.h"

iftImage* threshold_segmentation(iftImage* image, float min_obj){
	//Grayscale thresholding
	int threshold =  iftOtsu(image);
	iftImage* binary_global =  iftThreshold(image, 0, threshold, 1);

	//Grayscale thresholding
//	int first_threshold = (iftMaximumValue(image) + iftMinimumValue(image))/4;
//	iftImage* binary_global =  iftThreshold(image, first_threshold, 255, 1);
//	int threshold =  iftOtsuInRegion(image, binary_global);
//	iftDestroyImage(&binary_global);
//	binary_global = iftThreshold(image, 0, threshold, 1);

	iftImage *tmp;

	iftAdjRel *disk1 = iftCircular(1.0);
	iftAdjRel *disk5 = iftCircular(MAX(image->xsize,image->ysize)/200);

	//Hair removal by morphological opening
	tmp = binary_global;
	binary_global = iftOpen(binary_global, disk5);
	iftDestroyImage(&tmp);

	//Labels each component
	iftImage *labels = iftFastLabelComp(binary_global, disk1);

	iftVoxel voxel; voxel.x = 3; voxel.y = 3; voxel.z = 0;
	int label1 = labels->val[iftGetVoxelIndex(labels,voxel)];

	voxel.x = labels->xsize - 3; voxel.y = 3; voxel.z = 0;
	int label2 = labels->val[iftGetVoxelIndex(labels,voxel)];

	voxel.x = 3; voxel.y = labels->ysize - 3; voxel.z = 0;
	int label3 = labels->val[iftGetVoxelIndex(labels,voxel)];

	voxel.x = labels->xsize -3; voxel.y = labels->ysize -3; voxel.z = 0;
	int label4 = labels->val[iftGetVoxelIndex(labels,voxel)];

	//Remove black frame and take the complement
	for(int p = 0; p < image->n; p++)
		binary_global->val[p] = ! (binary_global->val[p] & !( (labels->val[p] == label1) | (labels->val[p] == label2) | (labels->val[p] == label3) | (labels->val[p] == label4) ) );

	//Find background component
	iftDestroyImage(&labels);
	labels = iftFastLabelComp(binary_global, disk1);

	voxel.x = 3; voxel.y = 3; voxel.z = 0;
	label1 = labels->val[iftGetVoxelIndex(labels,voxel)];

	voxel.x = labels->xsize - 3; voxel.y = 3; voxel.z = 0;
	label2 = labels->val[iftGetVoxelIndex(labels,voxel)];

	voxel.x = 3; voxel.y = labels->ysize - 3; voxel.z = 0;
	label3 = labels->val[iftGetVoxelIndex(labels,voxel)];

	voxel.x = labels->xsize -3; voxel.y = labels->ysize -3; voxel.z = 0;
	label4 = labels->val[iftGetVoxelIndex(labels,voxel)];

	for(int p = 0; p < image->n; p++)
		binary_global->val[p] = !( (labels->val[p] == label1) | (labels->val[p] == label2) | (labels->val[p] == label3) | (labels->val[p] == label4));

	//Eliminate small areas
	tmp = binary_global;
	binary_global = iftSelectCompAboveArea(binary_global, disk1,image->n * min_obj);
	iftDestroyImage(&tmp);

	//Optional: dilation
	//iftAdjRel *disk10 = iftCircular(10);
	//tmp = binary_global;
	//binary_global = iftDilate(binary_global, disk10);
	//iftDestroyAdjRel(&disk10);
	//iftDestroyImage(&tmp);

	iftDestroyImage(&labels);
	iftDestroyAdjRel(&disk1);
	iftDestroyAdjRel(&disk5);

	return binary_global;
}

void fix_slash(char *directory_path){
	if(directory_path[strlen(directory_path) -1 ] != '/' )
		strcat(directory_path, "/");
}

void create_dirs(char **argv){
	char buffer[256];

	DIR *dir = opendir(argv[1]);
	if (dir == NULL)
		iftError("Could not open input directory","lesion_seg_th");
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
	if(argc < 3)
		iftError("Usage: lesion_seg DATASET OUTPUT_FOLDER [MIN_OBJ]","lesion_seg");

	float min_obj = 0.01;

	if(argc > 3)
		min_obj = atof(argv[3]);

	create_dirs(argv);

	char input_dir[256];
	strcpy(input_dir, argv[1]);
	fix_slash(input_dir);
	strcat(input_dir, "orig/");

	char output_vis[256];
	strcpy(output_vis, argv[2]);
	fix_slash(output_vis);
	strcat(output_vis, "vis/");

	char output_label[256];
	strcpy(output_label, argv[2]);
	fix_slash(output_label);
	strcat(output_label, "labels/");

	int nimages = iftCountImageNames(input_dir, "ppm");
	iftImageNames *img_names = iftCreateAndLoadImageNames(nimages, input_dir, "ppm");

	iftColor green;
	green.val[0] = 0; green.val[1] = 0; green.val[2] = 255;
	green = iftRGBtoYCbCr(green);

	iftAdjRel *adj = iftCircular(1.5);
	iftAdjRel *adj0 = iftCircular(0.0);

	char name_buffer[256];
	for(int i = 0; i < nimages; i++){
		strcpy(name_buffer, input_dir);
		strcat(name_buffer, img_names[i].image_name);

		printf("Processing image %s.\n",name_buffer);

		iftImage *image = iftReadImageP6(name_buffer);

		//Thresholding
		iftImage *label = threshold_segmentation(image, min_obj);

		//Outputs thresholding results
		iftImage* drawn_labels = iftCopyImage(image);
		iftDrawBorders(drawn_labels,label,adj,green,adj0);

		char output_buffer[256];
		strcpy(output_buffer, output_vis);
		strcat(output_buffer, img_names[i].image_name);
		iftWriteImageP6(drawn_labels, output_buffer);

		char *base_name = iftSplitStringAt(img_names[i].image_name, ".ppm",0);

		strcpy(output_buffer, output_label);
		strcat(output_buffer, base_name);
		strcat(output_buffer, ".pgm");
		iftWriteImageP5(label, output_buffer);

		free(base_name);
		iftDestroyImage(&drawn_labels);
		iftDestroyImage(&image);
		iftDestroyImage(&label);
	}

	iftDestroyAdjRel(&adj);
	iftDestroyAdjRel(&adj0);
	iftDestroyImageNames(img_names);

	return 0;
}


