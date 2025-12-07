

#include "ift.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libgen.h>

//Define for debugging
//#define IFT_CONVNET_DEBUG_WRITE_ALL

#define _SILENCE

#define PGM "pgm"
#define PPM "ppm"

/************* HEADERS *************/
int iftCompareImageName(  void * a,   void * b);

void validateInputs(char *archIn, char* archOut, char *image_directory, char *extension, int output_format, char* output_name);

void initializeVariables(char* arch, char *image_directory, char* extension, iftConvNetwork **convnet,
						   int *nimages, iftImageNames **img_names);
void processConvNetToMImage(char* directory_path, char *extension, char* output_path, char *image_name,
							   iftConvNetwork *convnet);
void processConvNetToDataSet(int it, char* directory_path, char *extension, iftImageNames *img_names,
							   iftConvNetwork *convnet, iftDataSet **Z);
/**********************************/


int main(int argc, char **argv) {
	if (argc != 8)
		iftError("Please provide the following parameters:\n<architecture.convnet> <NULL|architecture.out.convnet> <image_directory> <ppm|pgm>" \
				"<output_format: [0 = multiband_images, 1 = datasets]> <directory location|dataset file> <NTHREADS> \n\n", "main");

	char archIn[100];            strcpy(archIn , argv[1]);
	char archOut[100];           strcpy(archOut, argv[2]);
	char image_directory[100];   strcpy(image_directory, argv[3]);
	char extension[100];         strcpy(extension,argv[4]);
	int  output_format         = atoi(argv[5]);
	char output_name[100];       strcpy(output_name,argv[6]);


	validateInputs(archIn, archOut, image_directory, extension, output_format, output_name);

	iftConvNetwork *convnet;
	int nimages;
	iftImageNames *img_names;
	iftImage *img;
	char filename[200];
	timer *t1, *t2;


	initializeVariables(archIn, image_directory, extension, &convnet, &nimages, &img_names);

	omp_set_num_threads(atoi(argv[7]));

	t1 = iftTic();
	if (!output_format) {
		iftCreateOrCleanDirectory(output_name);

#ifndef _SILENCE
		puts("***** Processing Images *****");
#endif
#pragma omp parallel for shared(image_directory,img_names,convnet)
		for(int i = 0; i < nimages ; i++) {
#ifndef _SILENCE
			printf("It. = %d, Image %s, class %d\n", i, img_names[i].image_name, img_names[i].attribute);
#endif
			processConvNetToMImage(image_directory, extension, output_name, img_names[i].image_name, convnet);
		}

	}
	else {
		iftDataSet *Z;
		// iftCreateOrCleanFilesInDirectory("outputs/datasets/", "", ".data");

		// Calculate the number of features of a feature vector
		int xsize[convnet->nstages], ysize[convnet->nstages], zsize[convnet->nstages], nbands[convnet->nstages];
		iftImageDimensionsAlongNetwork(convnet, xsize, ysize, zsize, nbands);
		int nfeatures = xsize[convnet->nstages - 1] * ysize[convnet->nstages - 1] *
						zsize[convnet->nstages - 1] * nbands[convnet->nstages - 1];

		Z = iftCreateDataSet(nimages, nfeatures);
#ifndef _SILENCE
		printf("Number of features: %d\n", nfeatures);

		puts("\n***** Processing Images *****");
#endif
		int t=0;
#pragma omp parallel for shared(image_directory,img_names,convnet,Z,t)
		for(int i = 0; i < nimages ; i++) {
#ifndef _SILENCE
#pragma omp critical 
		  t = t + 1;
		  printf("It. = %6d/%6d(%6.2f), Image %s, class %d\n",t,nimages,(float)t/nimages*100., img_names[i].image_name, img_names[i].attribute);
#endif
		  processConvNetToDataSet(i, image_directory, extension, img_names, convnet, &Z);
		}
		Z->nclasses = iftCountNumberOfClasses(img_names, nimages);
		iftWriteOPFDataSet(Z, output_name);
	}
	t2 = iftToc();
#ifndef _SILENCE
	fprintf(stdout, "\n-> Time elapsed %f sec\n", iftCompTime(t1, t2)/1000);
#endif


	// Saving the convnet
	if (strcmp(archOut,"NULL") != 0) {
	  //iftCreateOrCleanDirectory("outputs/convnets");
	  convnet->with_weights = 1;
	  iftWriteConvNetwork(convnet, archOut);
	}

	// Deallocators
	iftDestroyImageNames(img_names);
	iftDestroyConvNetwork(&convnet);

	return -1;
}



/************* BODIES *************/
int iftCompareImageName(  void * a,   void * b) {
	iftImageNames *first = (iftImageNames*) a;
	iftImageNames *second = (iftImageNames*) b;
	if ( strcmp(first->image_name, second->image_name) < 0 ) return -1;
	if ( strcmp(first->image_name, second->image_name) == 0 ) return 0;

	return 1;
}


void validateInputs(char *archIn, char* archOut, char *image_directory, char* extension, int output_format, char* output_name) {
	char msg[100];

	if(!iftEndsWith(archIn, ".convnet")) {
	  sprintf(msg, "Error: file %s does't match the format '.convnet'", archIn);
	  iftError(msg, "validInputs");
	}

	if (strcmp(archOut,"NULL") != 0) {
	  if (!iftEndsWith(archOut, ".convnet")) {
	    sprintf(msg, "Error: file %s does't match the format '.convnet'", archOut);
	    iftError(msg, "validInputs");
	  }

	  char tmp[100];strcpy(tmp,archOut);
	  if (!iftDirExists(dirname(tmp))) {
	    sprintf(msg, "Error: the directory indicated by %s does not exist", archOut);
	    iftError(msg, "validInputs");
	  }
	}

	if (!iftDirExists(image_directory))
	  iftError("Error: Image directory doesn't exist!!!", "validInputs");

	if ( !(strcmp(extension, PGM)==0) && !(strcmp(extension,PPM)==0) ) {
	  sprintf(msg, "Error: extension format (%s) invalid'", extension);
	  iftError(msg, "validInputs");
	}

	if (output_format == 0) {
	  if (!iftDirExists(output_name))
	    iftError("Error: Output directory doesn't exist!!!", "validInputs");
	} else if (output_format == 1) {
	  if (!iftEndsWith(output_name, ".data")) {
	    sprintf(msg, "Error: file %s does't match the format '.data'", output_name);
	    iftError(msg, "validInputs");
	  }

	  char tmp[100];strcpy(tmp,output_name);
	  if (!iftDirExists(dirname(tmp)) )  {
	    sprintf(msg, "Error: the directory indicated by %s does not exist", output_name);
	    iftError(msg, "validInputs");
	  }
	} else {
	  iftError("output_format invalid! Try 0 or 1", "validInputs");
	}

}


  void initializeVariables(char* arch, char *image_directory, char* extension, iftConvNetwork **convnet,
						   int *nimages, iftImageNames **img_names) {
	char ext[10];
	char *pos;
	int i;

	pos = strrchr(arch, '.') + 1;
	sscanf(pos, "%s", ext);

	if (strcmp(ext, "convnet") == 0)
		*convnet = iftReadConvNetwork(arch);
	else {
		printf("Invalid file format: %s\n", ext);
		exit(-1);
	}

	// insert '/' in the end of the path, if it doesn't have it
	if (image_directory[strlen(image_directory)-1] != '/')
		strcat(image_directory, "/");

 	if ( (strcmp(extension, PGM)==0) ) {
	  *nimages = iftCountImageNames(image_directory, PGM);
	  *img_names = iftCreateAndLoadImageNames(*nimages, image_directory, PGM);
	}
 	if ( (strcmp(extension, PPM)==0) ) {
	  *nimages = iftCountImageNames(image_directory, PPM);
	  *img_names = iftCreateAndLoadImageNames(*nimages, image_directory, PPM);
	}
	//*img_names = iftCreateImageNames(*nimages);
	//iftLoadImageDirectory(*img_names, image_directory, PGM);

	qsort(*img_names, *nimages, sizeof(iftImageNames), iftCompareImageName);

#ifndef _SILENCE
	printf("Image Directory: %s\n", image_directory);
	printf("Number of Images: %d\n", *nimages);
	puts("");
#endif 
}

void processConvNetToMImage(char* directory_path, char *extension, char* output_path, char *image_name, iftConvNetwork *convnet) {
	iftMImage *input, *output;
	iftImage *img;
	char filename[300];

	sprintf(filename, "%s%s", directory_path, image_name);

	img = iftReadImageP5(filename);

	input  = iftImageToMImage(img, RGB_CSPACE);
	iftDestroyImage(&img);

	output = iftApplyConvNetwork(input, convnet);

//	iftWriteMImageBands(output, "a");
	// remove the extension from the image
	if ( strcmp(extension,PGM) == 0 )
	  strcpy(image_name, iftSplitStringOld(image_name, ".pgm", 0));
	if ( strcmp(extension,PPM) == 0 )
	  strcpy(image_name, iftSplitStringOld(image_name, ".ppm", 0));

	sprintf(filename, "%s/%s.mig", output_path, image_name);
	iftWriteMImage(output, filename);

	iftDestroyMImage(&input);
	iftDestroyMImage(&output);
}


void processConvNetToDataSet(int it, char* directory_path, char *extension, iftImageNames *img_names,
							  iftConvNetwork *convnet, iftDataSet **Z) {
	iftMImage *input, *output;
	iftImage *img;
	char filename[300];

	sprintf(filename, "%s%s", directory_path, img_names[it].image_name);


	if ( strcmp(extension,PGM) == 0) {
	  img = iftReadImageP5(filename);
	  if (img->maxval <= 255 )
	    input  = iftImageToMImage(img, GRAY_CSPACE);
	  else
	    input  = iftImageToMImage(img, GRAY_CSPACE); // to be implemented
	}
	if ( strcmp(extension,PPM) == 0) {
	  img = iftReadImageP6(filename);
	  input  = iftImageToMImage(img, RGB_CSPACE);
	  //input  = iftImageToMImage(img, YCbCr_CSPACE);
	  //input  = iftImageToMImage(img, WEIGHTED_YCbCr_CSPACE);
	}

//	iftMultMImageByScalar(input,1./255);
	iftDestroyImage(&img);

	output = iftApplyConvNetwork(input, convnet);

	iftFeatures *features = iftMImageToFeatures(output);
	(*Z)->sample[it].class = img_names[it].attribute;
	free((*Z)->sample[it].feat);
	(*Z)->sample[it].feat = features->val;

	iftDestroyMImage(&input);
	iftDestroyMImage(&output);
}
