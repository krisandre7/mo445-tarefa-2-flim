#include "ift.h"

/**
 * Create n convnets (n = nclasses) with just the last layer random kernel bank different
 */
int main(int argc, char **argv) {
	if (argc != 4)
		iftError("Please provide the following parameters:\n <arch_model.convnet> <nclasses> <output_directory>\n\n", "main");

	iftRandomSeed(IFT_RANDOM_SEED); // IFT_RANDOM_SEED is a fixed number - reproducibility
//	iftRandomSeed(time(NULL)); // really random

	char arch[200]; strcpy(arch, argv[1]);
	int nclasses = atoi(argv[2]);
	char output_dir[200]; strcpy(output_dir, argv[3]);

	if (!iftEndsWith(arch, ".convnet")) {
		char msg[200];
		sprintf(msg, "Error: file %s does't match the format '.convnet'", arch);
		iftError(msg, "validInputs");
	}

	// insert '/' in the end of the path, if it doesn't have it
	if (output_dir[strlen(output_dir)-1] != '/')
		strcat(output_dir, "/");

	iftCreateOrCleanDirectory(output_dir);

	iftConvNetwork *convnet = iftReadConvNetwork(arch);
	iftAdjRel *A = iftCopyAdjacency(convnet->k_bank[convnet->nlayers-1]->A);
	int nbands;

	if (convnet->nlayers > 1)
		nbands = convnet->nkernels[convnet->nlayers-2];
	else nbands = 1;

	int nkernels = convnet->nkernels[convnet->nlayers-1];
	iftDestroyConvNetwork(&convnet);

	printf("Convnet: %s\n", arch);
	printf("nclasses = %d\n", nclasses);
	printf("Output Directory = %s\n", output_dir);
	printf("A->n = %d\n", A->n);
	printf("nbands = %d\n", nbands);
	printf("nkernels = %d\n", nkernels);

	convnet = iftReadConvNetwork(arch);
	convnet->with_weights = 1;
	for (int i = 0; i < nclasses; i++) {
		iftDestroyMMKernel(&convnet->k_bank[convnet->nlayers-1]); // Destroy the last layer filter bank
		char filename[300];
		char *name1, *name2;
		name1 = iftSplitStringAt(arch, ".convnet", 0);
		name2 = iftSplitStringAt(name1, "/", -1);
		sprintf(filename, "%s%s.%.4d.convnet", output_dir, name2, i+1);

		convnet->k_bank[convnet->nlayers-1] = iftRandomMMKernel(A, nbands, nkernels);
		iftWriteConvNetwork(convnet, filename);
		puts(filename);
		free(name1);
		free(name2);
	}

	free(convnet);


	return -1;
}
