#include "ift.h"

void iftMIGsToDataSet(char *mig_dir, iftConvNetwork *convnet, char *dataset_name);

// Transform a mimage set in a dataset
int main(int argc, char **argv) {
	if (argc != 4)
		iftError(
				"Usage: iftMIGsToDataSet <mig_directory> <arch.convnet> <dataset.data>\n\n",
				"main");

	if (!iftEndsWith(argv[2], ".convnet")) {
		char msg[200];
		sprintf(msg, "Error: file %s does't match the format '.convnet'",
				argv[2]);
		iftError(msg, "validInputs");
	}

//	iftRandomSeed(IFT_RANDOM_SEED);

	char mig_dir[200];
	strcpy(mig_dir, argv[1]);
	iftConvNetwork *convnet = iftReadConvNetwork(argv[2]);
	char dataset_name[200];
	strcpy(dataset_name, argv[3]);

	// insert '/' in the end of the path, if it doesn't have it
	if (mig_dir[strlen(mig_dir) - 1] != '/')
		strcat(mig_dir, "/");

	printf("MImage Directory: %s\n", mig_dir);
	printf("Convnet: %s\n", argv[2]);
	printf("Output: %s\n", dataset_name);

	iftMIGsToDataSet(mig_dir, convnet, dataset_name);

	iftDestroyConvNetwork(&convnet);

	puts("\nDone...\n");

	return -1;
}



void iftMIGsToDataSet(char *mig_dir, iftConvNetwork *convnet, char *dataset_name) {
	int nimages;
	iftImageNames *img_names;
	iftDataSet *Z;

	int xsize[convnet->nstages], ysize[convnet->nstages],
			zsize[convnet->nstages], nbands[convnet->nstages];
	iftImageDimensionsAlongNetwork(convnet, xsize, ysize, zsize, nbands);
	int nfeatures = xsize[convnet->nstages - 1] * ysize[convnet->nstages - 1]
			* zsize[convnet->nstages - 1] * nbands[convnet->nstages - 1];

	nimages = iftCountImageNames(mig_dir, "mig");
	img_names = iftCreateAndLoadImageNames(nimages, mig_dir, "mig");

	printf("nimages = %d\n", nimages);

	Z = iftCreateDataSet(nimages, nfeatures);

	for (int i = 0; i < nimages; i++) {
		char filename[200];
		sprintf(filename, "%s%s", mig_dir, img_names[i].image_name);
		printf("Processing %s\n", filename);

		iftMImage *mimg = iftReadMImage(filename);

		iftFeatures *features = iftMImageToFeatures(mimg);
		Z->sample[i].truelabel = img_names[i].attribute;
		free(Z->sample[i].feat);
		Z->sample[i].feat = features->val;

		iftDestroyMImage(&mimg);
	}

	Z->nclasses = iftCountNumberOfClasses(img_names, nimages);
	iftWriteOPFDataSet(Z, dataset_name);

	iftDestroyImageNames(img_names);
}
