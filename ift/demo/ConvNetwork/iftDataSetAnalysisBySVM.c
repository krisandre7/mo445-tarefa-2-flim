#include "ift.h"

enum { DEMO_LINEAR, DEMO_RBF, DEMO_PRECOMPUTED }; /* demo kernel enumeration */

/*
 * This program trains a SVM for a given training dataset and calculates the number of support
 * vectors and their classes from each class from the dataset.
 * The results are stored in an outuput_file.
 *
 * WARNING: The classes of the dataset must be contiguous, ie, if the dataset has 10 classes, the
 * classes must be from 1 to 10, because is calculated a histrogram of 1..Ztrain->nclasses
 */

int main(int argc, char **argv) {
	iftDataSet      *Z1[4] = {NULL, NULL, NULL, NULL}, *Ztrain = NULL;
	iftSVM          *svm = NULL;
	timer           *t1 = NULL, *t2 = NULL;
	int              i, num_of_comps, reduction;


	if (argc != 7)
		iftError("Usage: iftDataSetAnalysisBySVM <dataset.dat>"
						" <preprocess_type [0=NoReduction, 1=PCA, 2=SupPCA]>"
						" <num_of_comps>"
						" <kernel_type [0=linear,1=RBF,2=precomputed(linear)]"
						" <multiclass [0=OVO,1=OVA]> <output_file.txt>", "main");

	iftRandomSeed(IFT_RANDOM_SEED);

	/* Initialization */
	Z1[0] = iftReadOPFDataSet(argv[1]); // Read dataset Z
	iftSetStatus(Z1[0], IFT_TRAIN);
	printf("Total number of samples  %d\n", Z1[0]->nsamples);
	printf("Total number of features %d\n", Z1[0]->nfeats);
	printf("Total number of classes  %d\n", Z1[0]->nclasses);
	iftSetDistanceFunction(Z1[0], 1);

	reduction = atoi(argv[2]);
	num_of_comps = atoi(argv[3]);

	if ((num_of_comps <= 0) && (reduction > 0))
		iftError("Cannot reduce feature space to 0 or less components", "main");

	// SVM
	int kernel_type = atoi(argv[4]);
	float C = 1e5;
	float sigma = 0.1;

	t1 = iftTic();

	switch (kernel_type) {
	case DEMO_LINEAR:
		svm = iftCreateLinearSVC(C);
		break;
	case DEMO_RBF:
		svm = iftCreateRBFSVC(C, sigma);
		break;
	case DEMO_PRECOMPUTED:
		svm = iftCreatePreCompSVC(C);
		break;
	default:
		iftError("Invalid kernel type", "main");
	}


	switch (reduction) {
	case 0:
		puts("--- Normalizing the DataSet ---");
		Z1[2] = iftNormalizeDataSet(Z1[0]);
		iftDestroyDataSet(&Z1[0]);
		break;
	case 1:
		puts("--- Centralizing the DataSet ---");
		Z1[1] = iftCentralizeDataSet(Z1[0]);
		iftDestroyDataSet(&Z1[0]);
		Z1[2] = iftTransFeatSpaceByPCA(Z1[1], num_of_comps);
		iftDestroyDataSet(&Z1[1]);
		break;
	case 2:
		puts("--- Centralizing the DataSet ---");
		Z1[1] = iftCentralizeDataSet(Z1[0]);
		iftDestroyDataSet(&Z1[0]);
		Z1[2] = iftTransFeatSpaceBySupPCA(Z1[1], num_of_comps);
		iftDestroyDataSet(&Z1[1]);
		break;
	default:
		iftError("Invalid reduction option", "main");
	}


	if (kernel_type == DEMO_PRECOMPUTED) {
		uchar traceNormalize = 0;
		float ktrace;

		Z1[3] = iftKernelizeDataSet2(Z1[2], Z1[2], LINEAR, traceNormalize, 	&ktrace);
		Ztrain = Z1[3];
	} else {
		Ztrain = Z1[2];
	}

	if (atoi(argv[5]) == 0) { /* OVO */
		puts("--- Training SVM OVO ---");
		iftSVMTrainOVO(svm, Ztrain); // Training
	} else {
		puts("--- Training SVM OVA ---");
		iftSVMTrainOVA(svm, Ztrain);
	}


	FILE *fp = fopen(argv[6], "w");
	printf("- nmodels: %d\n", svm->nmodels);
	fprintf(fp, "Class, Number of Positive Support Vectors, Number of Negative Support Vectors, Number of Classes\n");
	for (int idxModel = 0; (idxModel < svm->nmodels) && (svm->model[idxModel] != NULL); idxModel++) {
		int *histogram = iftAllocIntArray(Ztrain->nclasses);
		int  pos_svs = 0; // number of positive vectors

		for(int sv = 0; sv < svm->model[idxModel]->l; sv++) {
			if ((svm->model[idxModel]->sv_coef[0][sv] != C) && (svm->model[idxModel]->sv_coef[0][sv] != 0)) {
				int idx_class = Ztrain->sample[svm->model[idxModel]->sv_indices[sv]-1].truelabel - 1;

				if (idx_class == idxModel)
					pos_svs++;
				histogram[idx_class]++;
			}
		}

		int nclasses = 0;

		for (int c = 0; c < Ztrain->nclasses; c++)
			if (histogram[c] != 0)
				nclasses++;

		free(histogram);

		printf("- Number of Classes: %d\n\n", nclasses);
		printf("--- Class: %d\n", idxModel+1);
		printf("- Number of Positive Support Vectors: %d/%d\n", pos_svs, Ztrain->nsamples);
		printf("- Number of Negative Support Vectors: %d/%d\n", svm->model[idxModel]->l - pos_svs, Ztrain->nsamples);
		printf("- Total of Support Vectors: %d/%d\n", svm->model[idxModel]->l, Ztrain->nsamples);
		fprintf(fp, "%d %d %d %d\n", idxModel+1, pos_svs, svm->model[idxModel]->l - pos_svs, nclasses);
	}
	fclose(fp);

	if (Z1[2] != NULL)
		iftDestroyDataSet(&Z1[2]);
	if (Z1[3] != NULL)
		iftDestroyDataSet(&Z1[3]);

	iftDestroySVM(svm);

	t2 = iftToc();

	fprintf(stdout, "-> Time elapsed: %f secs\n", iftCompTime(t1, t2)/1000);

	return 0;
}
