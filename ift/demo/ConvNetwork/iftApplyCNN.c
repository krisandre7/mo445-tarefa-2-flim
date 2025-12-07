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
#define PNG "png"

/************* HEADERS *************/
int iftCompareImageName(  void * a,   void * b);

void validateInputs(char *archIn, char* archOut, char *image_directory,
		char *extension, int output_format, char* output_name);

void initializeVariables(char* arch, char *image_directory, char* extension,
		iftConvNetwork **convnet, int *nimages, iftImageNames **img_names);
void processConvNetToMImage(char *img_name, char *ext,
		char *output_path, iftConvNetwork *convnet);
void processConvNetToDataSet(int it, char *img_name, int img_class, char *ext,
		iftConvNetwork *convnet, iftDataSet **Z);

void run_CNN1(char *archIn, char *archOut, FILE *fp, char *extension, int output_format, char *output_name);
void run_CNN2(char *archIn, char *archOut, char *image_base, char *extension, int output_format, char *output_name);
/**********************************/

int main(int argc, char **argv) {
	if (argc != 7)
        iftError(
                "Provide:\n <architecture.convnet> \n <NULL|architecture.out.convnet> \n" \
                "<image_base: [\".txt\" or \"image directory\"]> \n <ppm|pgm|mig> \n" \
                "<output_format: [0 = multiband_images, 1 = datasets]> \n <directory_location | dataset_file.data>",
                "main");

	char archIn[256];
	strcpy(archIn, argv[1]);
	char archOut[256];
	strcpy(archOut, argv[2]);
	char image_base[256];
	strcpy(image_base, argv[3]);
	char extension[256];
	strcpy(extension, argv[4]);
	int output_format = atoi(argv[5]);
	char output_name[256];
	strcpy(output_name, argv[6]);
	timer *t1, *t2;

//	iftRandomSeed(IFT_RANDOM_SEED); // IFT_RANDOM_SEED is a fixed number - reproducibility
	iftRandomSeed(time(NULL)); // really random

	t1 = iftTic();

	validateInputs(archIn, archOut, image_base, extension, output_format, output_name);

	printf("ArchIn: %s\n", archIn);
	printf("ArchOut: %s\n", archOut);
	printf("Image base: %s\n", image_base);
	printf("Output name: %s\n", output_name);

	if (iftEndsWith(image_base, "txt")) {
		FILE *fp = fopen(image_base, "r");
		if (fp == NULL) {
			char msg[256];
			sprintf(msg, "Can't open input file \"%s\"!\n", image_base);
            iftError(msg, "main");
		}
		run_CNN1(archIn, archOut, fp, extension, output_format, output_name);
		fclose(fp);
	} else
		run_CNN2(archIn, archOut, image_base, extension, output_format, output_name);


	t2 = iftToc();
	fprintf(stdout, "\n-> Time elapsed %f sec\n", iftCompTime(t1, t2) / 1000);

	return 0;
}

/************* BODIES *************/
int iftCompareImageName(  void * a,   void * b) {
	iftImageNames *first = (iftImageNames*) a;
	iftImageNames *second = (iftImageNames*) b;
	if (strcmp(first->image_name, second->image_name) < 0)
		return -1;
	if (strcmp(first->image_name, second->image_name) == 0)
		return 0;

	return 1;
}

void validateInputs(char *archIn, char* archOut, char *image_directory,
		char* extension, int output_format, char* output_name) {
	char msg[512];

	if (!iftEndsWith(archIn, ".convnet")) {
		sprintf(msg, "Error: file %s does't match the format '.convnet'",
				archIn);
        iftError(msg, "validInputs");
	}

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

	if (!iftEndsWith(image_directory, "txt") && (!iftDirExists(image_directory)))
        iftError("ERROR: The File is not .txt or the Image directory doesn't exist!!!", "validInputs");

	if (!(strcmp(extension, PGM) == 0) && !(strcmp(extension, PPM) == 0)
			&& !(strcmp(extension, MIG) == 0) && !(strcmp(extension, PNG) == 0)) {
		sprintf(msg, "Error: extension format (%s) invalid'", extension);
        iftError(msg, "validInputs");
	}

	if (output_format == 0) {
		if (!iftDirExists(output_name))
            iftError("Error: Output directory doesn't exist!!!", "validInputs");
	} else if (output_format == 1) {
		if (!iftEndsWith(output_name, ".data")) {
			sprintf(msg, "Error: file %s does't match the format '.data'",
					output_name);
            iftError(msg, "validInputs");
		}

		char tmp[512];
		strcpy(tmp, output_name);
		if (!iftDirExists(dirname(tmp))) {
			sprintf(msg, "Error: the directory indicated by %s does not exist",
					output_name);
            iftError(msg, "validInputs");
		}
	} else {
        iftError("output_format invalid! Try 0 or 1", "validInputs");
	}

}

void initializeVariables(char* arch, char *image_directory, char* extension,
		iftConvNetwork **convnet, int *nimages, iftImageNames **img_names) {

	*convnet = iftReadConvNetwork(arch);

	// insert '/' in the end of the path, if it doesn't have it
	if (image_directory[strlen(image_directory) - 1] != '/')
		strcat(image_directory, "/");

	if ((strcmp(extension, PGM) == 0)) {
		*nimages = iftCountImageNames(image_directory, PGM);
		*img_names = iftCreateAndLoadImageNames(*nimages, image_directory, PGM);
	}
	if ((strcmp(extension, PPM) == 0)) {
		*nimages = iftCountImageNames(image_directory, PPM);
		*img_names = iftCreateAndLoadImageNames(*nimages, image_directory, PPM);
	}
	if ((strcmp(extension, MIG) == 0)) {
		*nimages = iftCountImageNames(image_directory, MIG);
		*img_names = iftCreateAndLoadImageNames(*nimages, image_directory, MIG);
	}
	if ((strcmp(extension, PNG) == 0)) {
		*nimages = iftCountImageNames(image_directory, PNG);
		*img_names = iftCreateAndLoadImageNames(*nimages, image_directory, PNG);
	}

	// Concat the name of images with their directory
	char filename[512];
	for (int i = 0; i < *nimages; i++) {
		sprintf(filename, "%s%s", image_directory, (*img_names)[i].image_name);
		strcpy((*img_names)[i].image_name, filename);
	}

#ifndef _SILENCE
	printf("Image Directory: %s\n", image_directory);
	printf("Number of Images: %d\n", *nimages);
	puts("");
#endif 
}

void processConvNetToMImage(char *img_name, char *ext,
		char *output_path, iftConvNetwork *convnet) {
	iftMImage *input = NULL, *output = NULL;
	iftImage *img = NULL;

	puts(img_name);

	if ((strcmp(ext, PGM) == 0)) {
		img = iftReadImageP5(img_name);
		input = iftImageToMImage(img, GRAY_CSPACE);
		iftDestroyImage(&img);
	} else if (strcmp(ext, PPM) == 0) {
		img = iftReadImageP6(img_name);
		input = iftImageToMImage(img, RGB_CSPACE);
		iftDestroyImage(&img);
	} else if ((strcmp(ext, MIG) == 0))
		input = iftReadMImage(img_name);

	output = iftApplyConvNetwork(input, convnet);

	img_name = iftSplitStringAt(img_name, ".", -2);
	img_name = iftSplitStringAt(img_name, "/", -1);

	char filename[512];
	sprintf(filename, "%s%s.mig", output_path, img_name);
	iftWriteMImage(output, filename);

	iftDestroyMImage(&input);
	iftDestroyMImage(&output);
}

void processConvNetToDataSet(int it, char *img_name, int img_class, char *ext,
		iftConvNetwork *convnet, iftDataSet **Z) {
	iftMImage *input, *output;
	iftImage *img;

	if (strcmp(ext, PGM) == 0) {
		img   = iftReadImageP5(img_name);
		input = iftImageToMImage(img, GRAY_CSPACE);
		iftDestroyImage(&img);
	} else if (strcmp(ext, PPM) == 0) {
		img = iftReadImageP6(img_name);
		input = iftImageToMImage(img, RGB_CSPACE);
		iftDestroyImage(&img);
	} else if (strcmp(ext, MIG) == 0)
	if (strcmp(ext, MIG) == 0) {
		input = iftReadMImage(img_name);
	} else {
		img = iftReadImageByExt(img_name);
		if (!iftIsColorImage(img)) {
			input = iftImageToMImage(img, GRAY_CSPACE);
			iftDestroyImage(&img);
		} else {
			img = iftReadImageByExt(img_name);
			input = iftImageToMImage(img, RGB_CSPACE);
			iftDestroyImage(&img);
		}
	}

	output = iftApplyConvNetwork(input, convnet);

	iftFeatures *features = iftMImageToFeatures(output);
	(*Z)->sample[it].truelabel = img_class;
	free((*Z)->sample[it].feat);
	(*Z)->sample[it].feat = features->val;

	iftDestroyMImage(&input);
	iftDestroyMImage(&output);
}


void run_CNN1(char *archIn, char *archOut, FILE *fp, char *extension, int output_format, char *output_name) {
	iftConvNetwork *convnet = iftReadConvNetwork(archIn);
	char filename[512];
	int nimages, nclasses;

	if (fscanf(fp, "%d %d", &nimages, &nclasses) != 2)
        iftError("Read Error: nimages, nclasses", "run_CNN2");

	printf("Number of Images: %d\n", nimages);
	printf("Number of Classes: %d\n\n", nclasses);

	char **img_names = (char**) calloc(nimages, sizeof(char*));
	for (int i = 0; i < nimages; i++) {
		img_names[i] = iftAllocCharArray(512);

		if (fscanf(fp, "%s", img_names[i]) != 1)
            iftError("Read Error: img_name", "run_CNN2");
	}

	if (output_format == 0) {
		iftCreateOrCleanDirectory(output_name);
		puts("***** Processing Images to MImages *****");


		#pragma omp parallel for shared(img_names, extension, output_name, convnet) schedule(dynamic)
		for (int i = 0; i < nimages; i++) {
			char *str = iftSplitStringAt(img_names[i], "_", -2);
			int img_class = atoi(iftSplitStringAt(str, "/", -1));
			free(str);

			printf("It. = %d, Image %s, class %d\n", i, img_names[i], img_class);
			processConvNetToMImage(img_names[i], extension, output_name, convnet);
		}
	} else if (output_format == 1) {
		iftDataSet *Z;

		// Calculate the number of features of a feature vector
		int xsize[convnet->nstages], ysize[convnet->nstages],
				zsize[convnet->nstages], nbands[convnet->nstages];
		iftImageDimensionsAlongNetwork(convnet, xsize, ysize, zsize, nbands);
		int nfeatures = xsize[convnet->nstages - 1]
				* ysize[convnet->nstages - 1] * zsize[convnet->nstages - 1]
				* nbands[convnet->nstages - 1];

		Z = iftCreateDataSet(nimages, nfeatures);
		printf("Number of features: %d\n", nfeatures);

		puts("\n***** Processing Images to Dataset *****");

		#pragma omp parallel for shared(img_names, extension, convnet, Z) schedule(dynamic)
		for (int i = 0; i < nimages; i++) {
			char *str = iftSplitStringAt(img_names[i], "_", -2);
			int img_class = atoi(iftSplitStringAt(str, "/", -1));
			free(str);

			printf("It. = %d, Image %s, class %d\n", i, img_names[i], img_class);
			processConvNetToDataSet(i, img_names[i], img_class, extension, convnet, &Z);
		}
		Z->nclasses = nclasses;
		iftWriteOPFDataSet(Z, output_name);
	}

	// Saving the convnet
	if (strcmp(archOut, "NULL") != 0) {
		//iftCreateOrCleanDirectory("outputs/convnets");
		convnet->with_weights = 1;
		iftWriteConvNetwork(convnet, archOut);
	}
}


void run_CNN2(char *archIn, char *archOut, char *image_base, char *extension, int output_format, char *output_name) {
	iftConvNetwork *convnet;
	int nimages;
	iftImageNames *img_names;
	char filename[512];

	initializeVariables(archIn, image_base, extension, &convnet, &nimages, &img_names);

	if (output_format == 0) {
		iftCreateOrCleanDirectory(output_name);

		puts("***** Processing Images to MImages *****");

		#pragma omp parallel for shared(img_names, extension, output_name, convnet) schedule(dynamic)
		for (int i = 0; i < nimages; i++) {
#ifndef _SILENCE
			printf("It. = %d, Image %s, class %d\n", i, img_names[i].image_name, img_names[i].attribute);
#endif
			processConvNetToMImage(img_names[i].image_name, extension, output_name, convnet);
		}

	} else if (output_format == 1) {
		iftDataSet *Z;

		// Calculate the number of features of a feature vector
		int xsize[convnet->nstages], ysize[convnet->nstages],
				zsize[convnet->nstages], nbands[convnet->nstages];
		iftImageDimensionsAlongNetwork(convnet, xsize, ysize, zsize, nbands);
		int nfeatures = xsize[convnet->nstages - 1]
				* ysize[convnet->nstages - 1] * zsize[convnet->nstages - 1]
				* nbands[convnet->nstages - 1];

		Z = iftCreateDataSet(nimages, nfeatures);
		printf("Number of features: %d\n", nfeatures);

		puts("\n***** Processing Images to Dataset *****");

		#pragma omp parallel for shared(extension, img_names, convnet, Z) schedule(dynamic)
		for (int i = 0; i < nimages; i++) {
#ifndef _SILENCE
			printf("It. = %d, Image %s, class %d\n", i, img_names[i].image_name, img_names[i].attribute);
#endif
			processConvNetToDataSet(i, img_names[i].image_name, img_names[i].attribute, extension, convnet, &Z);
		}
		Z->nclasses = iftCountNumberOfClasses(img_names, nimages);
		iftWriteOPFDataSet(Z, output_name);
	}


	// Saving the convnet
	if (strcmp(archOut, "NULL") != 0) {
		//iftCreateOrCleanDirectory("outputs/convnets");
		convnet->with_weights = 1;
		iftWriteConvNetwork(convnet, archOut);
	}

	// Deallocators
	iftDestroyImageNames(img_names);
	iftDestroyConvNetwork(&convnet);
}






