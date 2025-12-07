#include "ift.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <libgen.h>


void validateInputs(char *arch1, char *arch2);


int main(int argc, char **argv) {
	if (argc != 4)
		iftError("Provide: <arch_1.convnet> <arch_2.convnet> <output_convnet_filename>", "main");

	char arch1[200]; strcpy(arch1, argv[1]);
	char arch2[200]; strcpy(arch2, argv[2]);
	char out_arch_name[200]; strcpy(out_arch_name, argv[3]);

	validateInputs(arch1, arch2);

	printf("** Convnet 1 : %s\n", arch1);
	printf("** Convnet 2 : %s\n", arch2);
	printf("** Output: %s\n", out_arch_name);

	iftConvNetwork *convnet1 = iftReadConvNetwork(arch1);
	iftConvNetwork *convnet2 = iftReadConvNetwork(arch2);

	iftConvNetwork *mconvnet = iftMergeConvNetwork(convnet1, convnet2);

//	printf("--- antes: %f\n--- depois: %f\n\n", convnet1->k_bank[0]->weight[1][10].val[10], mconvnet->k_bank[0]->weight[1][10].val[10]);
//	printf("--- antes: %f\n--- depois: %f\n\n", convnet1->k_bank[1]->weight[1][10].val[10], mconvnet->k_bank[1]->weight[1][10].val[10]);
//	printf("--- antes: %f\n--- depois: %f\n\n", convnet2->k_bank[0]->weight[2][0].val[10], mconvnet->k_bank[2]->weight[2][0].val[10]);

	printf("#### Saving the convnet: \"%s\" ####\n", out_arch_name);
	iftWriteConvNetwork(mconvnet, out_arch_name);

	iftDestroyConvNetwork(&convnet1);
	iftDestroyConvNetwork(&convnet2);
	iftDestroyConvNetwork(&mconvnet);

	return -1;
}



//******* BODY FUNCTIONS ************
void validateInputs(char *arch1, char *arch2) {
	char msg[200];

	if (!iftEndsWith(arch1, ".convnet")) {
		sprintf(msg, "Error: file %s does't match the format '.convnet'", arch1);
		iftError(msg, "validInputs");
	}

	if (!iftEndsWith(arch2, ".convnet")) {
		sprintf(msg, "Error: file %s does't match the format '.convnet'", arch2);
		iftError(msg, "validInputs");
	}
}
