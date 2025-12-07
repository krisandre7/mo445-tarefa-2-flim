#include "ift.h"

#define FEATS "feats"

void validate_inputs(char *feat_base, char *output);
void run_build_dataset1(FILE *fp, char *output);
void run_build_dataset2(char *feat_base, char *output);




int main(int argc, char **argv) {
	if (argc != 3)
			iftError("iftBuildDataSetFromFeats <feats_base>: [\".txt\" or \"feat_directory\"]>\n<output.data>", "main");

	char feat_base[512]; strcpy(feat_base, argv[1]);
	char output[512]; strcpy(output, argv[2]);
	timer *t1 = iftTic();

	validate_inputs(feat_base, output);

	if (iftEndsWith(feat_base, ".txt")) {
		FILE *fp = fopen(feat_base, "r");
		if (fp == NULL) {
			char msg[256];
			sprintf(msg, "Can't open input file \"%s\"!\n", feat_base);
			iftError(msg, "main");
		}
		run_build_dataset1(fp, output);
		fclose(fp);
	} else
		run_build_dataset2(feat_base, output);

	fprintf(stdout, "\n-> Time elapsed %f sec\n", iftCompTime(t1, iftToc()) / 1000);
	printf("--------------------------------------------------------\n\n");

	return 0;
}



/*********** BODIES ***********/
void validate_inputs(char *feat_base, char *output) {
	char msg[512];

	if ((!iftEndsWith(feat_base, ".txt")) && (!iftDirExists(feat_base))) {
		sprintf(msg, "Invalid Feature Base: \"%s\"... Try *.txt or a Directory", feat_base);
		iftError(msg, "validate_inputs");
	}

	if (!iftEndsWith(output, ".data")) {
		sprintf(msg, "Invalid Output Dataset: \"%s\"... Try *.data ", feat_base);
		iftError(msg, "validate_inputs");
	}

	printf("Feature Base: %s\n", feat_base);
	printf("Output: %s\n\n", output);
}


void run_build_dataset1(FILE *fp, char *output) {
	char filename[512], *str, *str2;
	int nsamples, nclasses;

	if (fscanf(fp, "%d %d", &nsamples, &nclasses) != 2)
		iftError("Read Error: nimages, nclasses", "run_build_dataset1");

	if (nsamples <= 0)
		iftError("Number of Samples is <= 0", "run_build_dataset1");

	printf("Number of Images/Samples: %d\n", nsamples);
	printf("Number of Classes: %d\n\n", nclasses);

	char **sample_names = (char**) calloc(nsamples, sizeof(char*));
	for (int i = 0; i < nsamples; i++) {
		sample_names[i] = iftAllocCharArray(512);

		if (fscanf(fp, "%s", sample_names[i]) != 1)
			iftError("Read Error: img_name", "run_build_dataset1");
	}

	iftFeatures *feat = iftReadFeatures2(sample_names[0]);
	iftDataSet *Z = iftCreateDataSet(nsamples, feat->n);
	Z->nclasses = nclasses;
	iftDestroyFeatures(&feat);
	iftFeatures **feats = (iftFeatures**) calloc(Z->nsamples, sizeof(iftFeatures*));

	puts("- Building Datasets from Feats");
	#pragma omp parallel for private(feat) schedule(dynamic)
	for (int i = 0; i < Z->nsamples; i++) {
		char *str  = iftSplitStringAt(sample_names[i], "_", -2);
		char *str2 = iftSplitStringAt(str, "/", -1);
		int class  = atoi(str2);

//		printf("[%d] - Class: %d - Feat Vector: %s\n", i, class, sample_names[i]);

		feats[i] = iftReadFeatures2(sample_names[i]);
		Z->sample[i].feat = feats[i]->val;
		Z->sample[i].truelabel = class;
		Z->sample[i].id = i;

		free(str);
		free(str2);
	}
	printf("Z->nclasses: %d, Z->nsamples: %d, Z->nfeats: %d\noutput: %s\n\n", Z->nclasses, Z->nsamples, Z->nfeats, output);
	iftWriteOPFDataSet(Z, output);

	iftDestroyDataSet(&Z);
	free(feats);
}


void run_build_dataset2(char *feat_base, char *output) {
	iftImageNames *feat_names;
	int nsamples;

	nsamples = iftCountImageNames(feat_base, FEATS);
	feat_names = iftCreateAndLoadImageNames(nsamples, feat_base, FEATS);
	char filename[512];
	sprintf(filename, "%s%s", feat_base, feat_names[0].image_name);
	iftFeatures *feat = iftReadFeatures2(filename);
	iftDataSet *Z = iftCreateDataSet(nsamples, feat->n);
	Z->nclasses = iftCountNumberOfClasses(feat_names, nsamples);
	iftDestroyFeatures(&feat);

	#pragma omp parallel for private(feat) schedule(dynamic)
	for (int i = 0; i < Z->nsamples; i++) {
		char filename[512];
		sprintf(filename, "%s%s", feat_base, feat_names[i].image_name);
		printf("[%d] - Class: %d - Feat Vector: %s\n", i, feat_names[i].attribute, filename);

		feat = iftReadFeatures2(filename);
		iftCopyFloatArray(Z->sample[i].feat, feat->val, feat->n);
		Z->sample[i].truelabel = feat_names[i].attribute;

		iftDestroyFeatures(&feat);
	}
	iftWriteOPFDataSet(Z, output);
	printf("Z->nclasses: %d, Z->nsamples: %d, Z->nfeats: %d\noutput: %s\n\n", Z->nclasses, Z->nsamples, Z->nfeats, output);

	iftDestroyDataSet(&Z);
	iftDestroyImageNames(feat_names);
}


























