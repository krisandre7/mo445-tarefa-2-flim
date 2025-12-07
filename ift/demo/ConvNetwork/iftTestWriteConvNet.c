#include "ift.h"



int main(int argc, char **argv) {
	if (argc != 2)
		iftError("Please provide the following parameters:\n<CONFIG_FILE.convnet>\n\n", "main");

	char *file_config = argv[1];
	char filename[200];
	iftConvNetwork *convnet,*convnet_r;
	filename[0] = 0;
	sprintf(filename, "output.convnet");
	// Read convnet
	convnet = iftReadConvNetwork(file_config);
	// Write convnet
	iftWriteConvNetwork(convnet,filename);
	// Read from the new file
	convnet_r = iftReadConvNetwork(filename);
	// Verify
	int diff = 0;
	for (int l=0; l < convnet->nlayers; l++) {
		for (int k=0; k < convnet->k_bank[l]->nkernels; k++) {
		  for (int b=0; b < convnet->k_bank[l]->nbands; b++) {
			  for(int j=0;j< convnet->k_bank[l]->A->n;j++){
			    if(convnet->k_bank[l]->weight[k][b].val[j] != convnet_r->k_bank[l]->weight[k][b].val[j]){
				  //printf("******* layer : %d, k: %d, b: %d, val: %f, read:%f \n",l,k,b,convnet->k_bank[l]->weight[k][b].val[j],convnet_r->k_bank[l]->weight[k][b].val[j]);
				  diff++;
			    }
			  }
		  }
		}
		if(diff == 0)
			printf("OK layer %d \n",l);
		else
			printf("Failed in layer %d \n",l);
	}
	if(diff == 0){
		printf("**** Ok ***** \n");
	}else{
		printf("*** #Diferences %d*****\n",diff);
	}
	// Destroy
	iftDestroyConvNetwork(&convnet);
	iftDestroyConvNetwork(&convnet_r);

	return -1;
}
