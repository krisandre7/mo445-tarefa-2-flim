#include "ift.h"




int main(int argc, char** argv) {
	if (argc != 3)
		iftError("Provide: <dataset.data> <class_value>", "main");

	iftDataSet *Z  = iftReadOPFDataSet(argv[1]);
	int class = atoi(argv[2]);

	for (int s; s < Z->nsamples; s++)
		Z->sample[s].truelabel = class;

	iftWriteOPFDataSet(Z, argv[1]);
	iftDestroyDataSet(&Z);

	return 0;
}
