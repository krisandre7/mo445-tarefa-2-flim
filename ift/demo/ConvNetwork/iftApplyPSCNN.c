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
#define CONVNET "convnet"

/************* HEADERS *************/
int iftCompareConNetworkNames(  void * a,   void * b);
void validateInputs(char *ps_arch_dir, char *image_directory, char *extension, int output_format, char* output_name);
void initializeVariables(char* ps_arch_dir, char *image_directory, char* extension, char ***convnets,
					int *narchs, int *nimages, iftImageNames **img_names);
iftImageNames *iftCreateAndLoadArchNames(int narchs, char *directory_path, char *file_extension);
void processConvNetToMImage(char* directory_path, char *extension, char* output_path, char *image_name,
							iftConvNetwork *convnet);
void processConvNetToDataSet(int it, char* directory_path, char *extension, iftImageNames *img_names,
							iftConvNetwork *convnet, iftDataSet **Z);

/**********************************/


int main(int argc, char **argv) {
	if (argc != 6)
		iftError("Please provide the following parameters:\n<Person-Specific_architecture_directory> <image_directory> <ppm|pgm|mig> " \
				"<output_format: [0 = multiband_images, 1 = datasets]> <directory_location|dataset_file>\n\n", "main");

	char ps_arch_dir[100]; 		strcpy(ps_arch_dir , argv[1]);
	char image_directory[100];	strcpy(image_directory, argv[2]);
	char extension[100];       	strcpy(extension,argv[3]);
	int  output_format	=		atoi(argv[4]);
	char output_name[100];      strcpy(output_name,argv[5]);

	validateInputs(ps_arch_dir, image_directory, extension, output_format, output_name);

	iftImageNames *img_names;
	char **convnet_names; // class i = [i-1]
	int nimages, narchs; //narchs = nclasses
	char filename[300];
	timer *t1, *t2;
	iftConvNetwork *convnet;

	iftRandomSeed(IFT_RANDOM_SEED); // IFT_RANDOM_SEED is a fixed number - reproducibility
//	iftRandomSeed(time(NULL)); // really random

	initializeVariables(ps_arch_dir, image_directory, extension, &convnet_names, &narchs, &nimages, &img_names);
	int nclasses = narchs;
	for (int i = 0; i < narchs; i++)
		puts(convnet_names[i]);

	t1 = iftTic();
	if (!output_format) {
		iftCreateOrCleanDirectory(output_name);

		puts("***** Processing Images *****");

		int i = 0; // iterator of img_names
		for (int c = 1; c <= nclasses; c++) { // for each class
			convnet = iftReadConvNetwork(convnet_names[c - 1]); // class c use the convnet_names[c-1]

			while ((i < nimages) && (img_names[i].attribute == c)) {
				printf("It. = %d, Image %s, class %d, convnet: %s\n", i,
						img_names[i].image_name, img_names[i].attribute,
						convnet_names[c - 1]);
				processConvNetToMImage(image_directory, extension, output_name,
						img_names[i].image_name, convnet);
				i++;
			}

			iftDestroyConvNetwork(&convnet);
		}
	}
	else {
		iftDataSet *Z;
		convnet = iftReadConvNetwork(convnet_names[0]); // just to obtain the measures

		// Calculate the number of features of a feature vector
		int xsize[convnet->nstages], ysize[convnet->nstages],
				zsize[convnet->nstages], nbands[convnet->nstages];
		iftImageDimensionsAlongNetwork(convnet, xsize, ysize, zsize, nbands);
		int nfeatures = xsize[convnet->nstages - 1]
				* ysize[convnet->nstages - 1] * zsize[convnet->nstages - 1]
				* nbands[convnet->nstages - 1];

		iftDestroyConvNetwork(&convnet);
		Z = iftCreateDataSet(nimages, nfeatures);
		printf("Number of features: %d\n", nfeatures);

		puts("\n***** Processing Images *****");

		int i = 0; // iterator of img_names
		int count; // number of images in each class
		for (int c = 1; c <= nclasses; c++) { // for each class
			convnet = iftReadConvNetwork(convnet_names[c - 1]); // class c use the convnet_names[c-1]

			count = 0;
			while ((i < nimages) && (img_names[i].attribute == c)) {
				printf("- It. = %d, Image %s, class %d, convnet: %s\n", i,
						img_names[i].image_name, img_names[i].attribute,
						convnet_names[c - 1]);
				processConvNetToDataSet(i, image_directory, extension,
						img_names, convnet, &Z);
				i++;
				count++;
			}
			printf("Class %d - %d images\n\n", c, count);

			iftDestroyConvNetwork(&convnet);
		}

		Z->nclasses = iftCountNumberOfClasses(img_names, nimages);
		iftWriteOPFDataSet(Z, output_name);
	}
	t2 = iftToc();
	fprintf(stdout, "\n-> Time elapsed %f sec\n", iftCompTime(t1, t2) / 1000);


	// Deallocators
	for (int i = 0; i < narchs; i++) {
		free(convnet_names[i]);
	}
	free(convnet_names);
	free(convnet);
	iftDestroyImageNames(img_names);

	return -1;
}



/************* BODIES *************/
int iftCompareConNetworkNames(  void * a,   void * b) {
    iftImageNames *first = (iftImageNames*) a;
    iftImageNames *second = (iftImageNames*) b;
    return strcmp(first->image_name, second->image_name);
}

void validateInputs(char *ps_arch_dir, char *image_directory, char *extension, int output_format, char* output_name) {
	char msg[100];

	if (!iftDirExists(ps_arch_dir)) {
		sprintf(msg, "Error: Person-Specific Architecture Directory: \"%s\" doesn't exist!!!", ps_arch_dir);
		iftError(msg, "validInputs");
	}

	if (!iftDirExists(image_directory)) {
		sprintf(msg, "Error: Image directory: \"%s\" doesn't exist!!!", image_directory);
		iftError(msg, "validInputs");
	}

	if (!(strcmp(extension, PGM) == 0) && !(strcmp(extension, PPM) == 0) && !(strcmp(extension, MIG) == 0)) {
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

		char tmp[100];
		strcpy(tmp, output_name);
		if (!iftDirExists(dirname(tmp))) {
			sprintf(msg, "Error: the directory indicated by %s does not exist", output_name);
			iftError(msg, "validInputs");
		}
	} else {
		iftError("output_format invalid! Try 0 or 1", "validInputs");
	}
}


void initializeVariables(char* ps_arch_dir, char *image_directory, char* extension, char ***convnets,
					int *narchs, int *nimages, iftImageNames **img_names) {

	iftImageNames *arch_names;
	char filename[300];

	// insert '/' in the end of the path, if it doesn't have it
	if (ps_arch_dir[strlen(ps_arch_dir)-1] != '/')
		strcat(ps_arch_dir, "/");

	*narchs = iftCountImageNames(ps_arch_dir, CONVNET);
	arch_names = iftCreateAndLoadArchNames(*narchs, ps_arch_dir, CONVNET);

	puts("");
	(*convnets) = (char**) calloc(*narchs, sizeof(char*));
	for (int i = 0; i < *narchs; i++) {
		sprintf(filename, "%s%s", ps_arch_dir, arch_names[i].image_name);
		(*convnets)[i] = iftAllocCharArray(100);
		strcpy((*convnets)[i], filename);
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
	if ( (strcmp(extension, MIG)==0) ) {
	  *nimages = iftCountImageNames(image_directory, MIG);
	  *img_names = iftCreateAndLoadImageNames(*nimages, image_directory, MIG);
	}
	printf("Image Directory: %s\n", image_directory);
	printf("Number of Images: %d\n", *nimages);
	printf("Person-Specific Architecture Directory: %s\n", ps_arch_dir);
	printf("Number of Archs: %d\n", *narchs);
	puts("");
}


/** Adaptation from the method: iftCreateAndLoadImageNames in iftParseInput
 * The adaptation was necessary to extract the class from the architectures names
 * that follow the following pattern: *****.0001.convnet, where ***** is anything and 0001 is the class
 *
 * http://pubs.opengroup.org/onlinepubs/007908799/xbd/re.html#tag_007_003_006
 * http://regexpal.com/
 * http://www.peope.net/old/regex.html
 * http://stackoverflow.com/questions/5696750/posix-regular-expressions-limit-repetitions
 */
iftImageNames *iftCreateAndLoadArchNames(int narchs, char *directory_path, char *file_extension) {
    int i = 0;
    DIR * directory_pointer;
    struct dirent * entry;
    iftImageNames *IN = iftCreateImageNames(narchs);
    char *class = NULL;
	regex_t regex;
	char exp[100]; sprintf(exp, "^.\\+\\.[0-9]\\+\\.%s$", file_extension);
	int reti;
	char msg[200];

	/* Compile regular expression */
	reti = regcomp(&regex, exp, 0);
	if (reti)
		iftError("Could not compile regex\n", "iftCreateAndLoadImageNames");

    directory_pointer = opendir(directory_path);
    if (!directory_path)
        iftError("Error opening directory path", "iftLoadDirectory");

    while ((entry = readdir(directory_pointer)) != NULL) {
    	reti = regexec(&regex, entry->d_name, 0, NULL, 0);
		if(!reti) { // Match
			strcpy(IN[i].image_name, entry->d_name);
			class = iftSplitStringAt(entry->d_name, ".", -2);
			IN[i].attribute = atoi(class);
			free(class);
			i++;
		}
		else if (reti == REG_NOMATCH) { // No Match
			printf("WARNING: %s doesn't match with the pattern %s\n", entry->d_name, exp);
		} else {
			regerror(reti, &regex, msg, sizeof(msg));
			sprintf(msg, "ERROR: Regex match failed: %s\n", msg);
			iftError(msg, "iftCreateAndLoadImageNames");
		}
    }

    qsort(IN, i, sizeof(iftImageNames), iftCompareConNetworkNames);
    regfree(&regex);
    closedir(directory_pointer);

    return IN;
}



void processConvNetToMImage(char* directory_path, char *extension, char* output_path, char *image_name, iftConvNetwork *convnet) {
	iftMImage *input = NULL, *output = NULL;
	iftImage *img = NULL;
	char filename[300];

	sprintf(filename, "%s%s", directory_path, image_name);

	if ((strcmp(extension, PGM)==0)) {
		img = iftReadImageP5(filename);

		input = iftImageToMImage(img, GRAY_CSPACE);
		iftDestroyImage(&img);
	}
	else if ( strcmp(extension, PPM) == 0) {
	  img = iftReadImageP6(filename);
	  input  = iftImageToMImage(img, RGB_CSPACE);
	  iftDestroyImage(&img);
	}
	else if ((strcmp(extension, MIG)==0))
		input = iftReadMImage(filename);

	output = iftApplyConvNetwork(input, convnet);

	// remove the extension from the image
	if ( strcmp(extension, PGM) == 0 )
	  strcpy(image_name, iftSplitStringOld(image_name, ".pgm", 0));
	if ( strcmp(extension, PPM) == 0 )
	  strcpy(image_name, iftSplitStringOld(image_name, ".ppm", 0));
	if ( strcmp(extension, MIG) == 0 )
	  strcpy(image_name, iftSplitStringOld(image_name, ".mig", 0));

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
	  iftDestroyImage(&img);
	}
	else if ( strcmp(extension,PPM) == 0) {
	  img = iftReadImageP6(filename);
	  input  = iftImageToMImage(img, RGB_CSPACE);
	  iftDestroyImage(&img);
	}
	else if ( strcmp(extension, MIG) == 0)
		input = iftReadMImage(filename);


	output = iftApplyConvNetwork(input, convnet);

	iftFeatures *features = iftMImageToFeatures(output);
	(*Z)->sample[it].truelabel = img_names[it].attribute;
	free((*Z)->sample[it].feat);
	(*Z)->sample[it].feat = features->val;

	iftDestroyMImage(&input);
	iftDestroyMImage(&output);
}
