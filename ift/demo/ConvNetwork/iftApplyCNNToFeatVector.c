#include "ift.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libgen.h>

//Define for debugging
//#define IFT_CONVNET_DEBUG_WRITE_ALL

//#define _SILENCE

#define PGM "pgm"
#define PPM "ppm"
#define MIG "mig"

/************* HEADERS *************/
int iftCompareImageName(  void * a,   void * b);

void validate_inputs(char *archIn, char* archOut, char *img_dir, char *ext, char* output_dir);

void initialize_variables(char* arch, char *img_dir, char* ext, char *output_dir,
		iftConvNetwork **convnet, int *nimages, iftImageNames **img_names);
void process_convnet(char *img_name, int class, char *ext, char *output_dir, iftConvNetwork *convnet);

void run_CNN1(char *archIn, char *archOut, FILE *fp, char *ext, char *output_dir);
void run_CNN2(char *archIn, char *archOut, char *img_base, char *ext, char *output_dir);
/**********************************/


/* Program that runs the CNN for each image in a directory or text file by saving the feature vector in a file .feats
 *
 * The Text File has the following organization:
 * 100 83 // number n of images, number of classes
 * ~/base/000001_0000001.pgm
 * ~/base/000001_0000002.pgm
 * ... // until n images
 */
int main(int argc, char **argv) {
	if (argc != 6)
		iftError(
				"iftApplyCNNToFeatVector <arch.convnet> <NULL|arch_out.convnet>\n" \
				"<image_base: [\".txt\" or \"image directory\"]> <ppm|pgm|mig>\n" \
				"<output_dir>", "main");

	char archIn[256];		strcpy(archIn, argv[1]);
	char archOut[256];		strcpy(archOut, argv[2]);
	char img_base[256];		strcpy(img_base, argv[3]);
	char ext[256]; 		   	strcpy(ext, argv[4]);
	char output_dir[256]; 	strcpy(output_dir, argv[5]);
	timer *t1, *t2;

//	iftRandomSeed(IFT_RANDOM_SEED); // IFT_RANDOM_SEED is a fixed number - reproducibility
	iftRandomSeed(time(NULL)); // really random

	t1 = iftTic();

	validate_inputs(archIn, archOut, img_base, ext, output_dir);

	printf("ArchIn: %s\n", archIn);
	printf("ArchOut: %s\n", archOut);
	printf("Image base: %s\n", img_base);
	printf("Output dir: %s\n", output_dir);

	if (iftEndsWith(img_base, "txt")) {
		FILE *fp = fopen(img_base, "r");
		if (fp == NULL) {
			char msg[256];
			sprintf(msg, "Can't open input file \"%s\"!\n", img_base);
		  	iftError(msg, "main");
		}
		run_CNN1(archIn, archOut, fp, ext, output_dir);
		fclose(fp);
	} else
		run_CNN2(archIn, archOut, img_base, ext, output_dir);

	t2 = iftToc();
	fprintf(stdout, "\n-> Time elapsed %f sec\n", iftCompTime(t1, t2) / 1000);

	return 0;
}

/************* BODIES *************/
void validate_inputs(char *archIn, char* archOut, char *img_dir, char* ext, char* output_dir) {
	char msg[512];

	if (!iftEndsWith(archIn, ".convnet")) {
		sprintf(msg, "Error: file %s does't match the format '.convnet'",
				archIn);
		iftError(msg, "validInputs");
	}

	if (!iftEndsWith(img_dir, "txt") && (!iftDirExists(img_dir)))
		iftError("ERROR: The File is not .txt or the Image directory doesn't exist!!!", "validInputs");

	if (!(strcmp(ext, PGM) == 0) && !(strcmp(ext, PPM) == 0)
			&& !(strcmp(ext, MIG) == 0)) {
		sprintf(msg, "Error: extension format (%s) invalid'", ext);
		iftError(msg, "validInputs");
	}

	if (!iftDirExists(output_dir))
		iftMakeDir(output_dir);

	if (strcmp(archOut, "NULL") != 0) {
		if (!iftEndsWith(archOut, ".convnet")) {
			sprintf(msg, "Error: file %s does't match the format '.convnet'",
					archOut);
			iftError(msg, "validInputs");
		}

		char tmp[512];
		strcpy(tmp, archOut);
		if (!iftDirExists(dirname(tmp))) {
			sprintf(msg, "Error: the directory indicated by %s does not exist",
					archOut);
			iftError(msg, "validInputs");
		}
	}
}

void initialize_variables(char* arch, char *img_dir, char* ext, char *output_dir,
		iftConvNetwork **convnet, int *nimages, iftImageNames **img_names) {

	*convnet = iftReadConvNetwork(arch);

	// insert '/' in the end of the path, if it doesn't have it
	if (img_dir[strlen(img_dir) - 1] != '/')
		strcat(img_dir, "/");

	// insert '/' in the end of the path, if it doesn't have it
	if (output_dir[strlen(output_dir) - 1] != '/')
		strcat(output_dir, "/");

	if ((strcmp(ext, PGM) == 0)) {
		*nimages = iftCountImageNames(img_dir, PGM);
		*img_names = iftCreateAndLoadImageNames(*nimages, img_dir, PGM);
	}
	if ((strcmp(ext, PPM) == 0)) {
		*nimages = iftCountImageNames(img_dir, PPM);
		*img_names = iftCreateAndLoadImageNames(*nimages, img_dir, PPM);
	}
	if ((strcmp(ext, MIG) == 0)) {
		*nimages = iftCountImageNames(img_dir, MIG);
		*img_names = iftCreateAndLoadImageNames(*nimages, img_dir, MIG);
	}

	// Concat the name of images with their directory
	char filename[512];
	for (int i = 0; i < *nimages; i++) {
		sprintf(filename, "%s%s", img_dir, (*img_names)[i].image_name);
		strcpy((*img_names)[i].image_name, filename);
	}

	printf("Number of Images: %d\n", *nimages);
	puts("");
}


void process_convnet(char *img_name, int class, char *ext, char *output_dir, iftConvNetwork *convnet) {
	iftMImage *input, *output;
	iftImage *img;
	char filename[512], *str, *str2, ext2[20];

	strcpy(ext2, ".");
	strcat(ext2, ext);

	if (strcmp(ext, PGM) == 0) {
		img = iftReadImageP5(img_name);
		input = iftImageToMImage(img, GRAY_CSPACE);
		iftDestroyImage(&img);
	} else if (strcmp(ext, PPM) == 0) {
		img = iftReadImageP6(img_name);
		input = iftImageToMImage(img, RGB_CSPACE);
		iftDestroyImage(&img);
	} else if (strcmp(ext, MIG) == 0)
		input = iftReadMImage(img_name);

	output = iftApplyConvNetwork(input, convnet);

	str = iftSplitStringAt(img_name, "/", -1);
	str2 = iftSplitStringAt(str, ext2, 0);
	strcat(str2, ".feats");
	strcpy(filename, output_dir);
	strcat(filename, str2);

	iftFeatures *features = iftMImageToFeatures(output);
	// printf("Class: %d\nImg: %s\nFeature Vector: %s\n\n", class, img_name, filename);
	iftWriteFeatures2(features, filename);

	iftDestroyMImage(&input);
	iftDestroyMImage(&output);
	free(str);
	free(str2);
	iftDestroyFeatures(&features);
}

void run_CNN1(char *archIn, char *archOut, FILE *fp, char *ext, char *output_dir) {
	iftConvNetwork *convnet = iftReadConvNetwork(archIn);
	char filename[512];
	int nimages, nclasses;

	if (fscanf(fp, "%d %d", &nimages, &nclasses) != 2)
		iftError("Read Error: nimages, nclasses", "run_CNN1");

	printf("Number of Images: %d\n", nimages);

	char **img_names = (char**) calloc(nimages, sizeof(char*));
	for (int i = 0; i < nimages; i++) {
		img_names[i] = iftAllocCharArray(512);

		if (fscanf(fp, "%s", img_names[i]) != 1)
			iftError("Read Error: img_name", "run_CNN1");

		char *str  = iftSplitStringAt(img_names[i], "_", -2);
		char *str2 = iftSplitStringAt(str, "/", -1);
		int img_class = atoi(str2);
		free(str);
		free(str2);

		printf("[%d/%d]\n", i+1, nimages);
		process_convnet(img_names[i], img_class, ext, output_dir, convnet);
	}

	// Saving the convnet
	if (strcmp(archOut, "NULL") != 0) {
		convnet->with_weights = 1;
		iftWriteConvNetwork(convnet, archOut);
	}
}


void run_CNN2(char *archIn, char *archOut, char *img_base, char *ext, char *output_dir) {
	iftConvNetwork *convnet;
	int nimages;
	iftImageNames *img_names;

	initialize_variables(archIn, img_base, ext, output_dir, &convnet, &nimages, &img_names);

	#pragma omp parallel for shared(img_names, ext, output_dir, convnet) schedule(dynamic)
	for (int i = 0; i < nimages; i++)
		process_convnet(img_names[i].image_name, img_names[i].attribute, ext, output_dir, convnet);

	// Saving the convnet
	if (strcmp(archOut, "NULL") != 0) {
		convnet->with_weights = 1;
		iftWriteConvNetwork(convnet, archOut);
	}

	// Deallocators
	iftDestroyImageNames(img_names);
	iftDestroyConvNetwork(&convnet);
}






















