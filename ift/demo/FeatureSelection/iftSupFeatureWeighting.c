#include "ift.h"

int fcompare (  void * a,   void * b)
{
  float fa = *(float*)a;
  float fb = *(float*)b;

  if(fa<fb)
  {
  	return -1;
  }
  else if(fa>fb)
  {
  	return 1;
  }
  else
  {
  	return 0;
  }
}

void sumFloatArray(float *array1, float* array2, int n)
{
	for (--n; n>=0; --n)
	{
		array1[n] += array2[n];
	}
}

void initFloatArray(float* array, float value, int n)
{
	for (--n; n>=0; --n)
	{
		array[n] = value;
	}
}

//The Z2 dataset is internally split from Z1 in the feature selection method
void splitDataSet(iftDataSet* Z, iftDataSet **Zlearn, iftDataSet **Zeval, iftDataSet **Ztest, float learn, float eval, float test)
{
	iftSelectSupTrainSamples(Z, learn+eval);
	*Ztest = Z;
	*Zeval = iftExtractSamples(Z, IFT_TRAIN);
	iftSelectSupTrainSamples(*Zeval, learn/(learn+eval));
	*Zlearn = iftExtractSamples(*Zeval, IFT_TRAIN);
}

int main(int argc, char *argv[])
{
	float z1Perc, z2Perc, z3Perc, z4Perc;
	int bootstraps = 10, i, j, n;
	float featsPenalty = 0.2;
	int nruns = 5;
	iftDataSet* Z;
	iftDataSet *Zlearn, *Ztest, *Zeval;
	iftFeatureSelector* featSelector;
	float *sumRank = NULL;

	iftDataSetMachineLearning classifier = NULL;
	iftDataSetEvalMetric metric = NULL;

	int classifierOption, metricOption;

	if(argc<9)
	{
		iftError("Usage: <dataset> <classifier: [0=OPF, 1=SVM]> <eval metric: [0=SkewedTruePositives, 1=TruePositives, 2=ClassifierAccuracy]> <Z1 percentage> <Z2 percentage> <Z3 percentage> <Z4 percentage>  <# of bootstraps> [<featsPenalty = 0.2> <# of fitness sampling = 5>]", "main");
		return 1;
	}

	printf("Reading Dataset %s ...\n", argv[1]);

	Z = iftReadOPFDataSet(argv[1]);
	n = Z->nfeats;

	sscanf(argv[2], "%d", &classifierOption);
	sscanf(argv[3], "%d", &metricOption);
	sscanf(argv[4], "%f", &z1Perc);
	sscanf(argv[5], "%f", &z2Perc);
	sscanf(argv[6], "%f", &z3Perc);
	sscanf(argv[7], "%f", &z4Perc);
	sscanf(argv[8], "%d", &bootstraps);
	if(argc>9)
		sscanf(argv[9], "%f", &featsPenalty);
	if(argc>10)
		sscanf(argv[10], "%d", &nruns);

	if(z1Perc+z2Perc+z3Perc+z4Perc!=1.0f)
	{
		iftError("Invalid arguments: z1+z2+z3+z4!=1.0f", "main");
		return 1;
	}

	switch(classifierOption)
	{
		case 0:
			classifier = iftFeatSelOPFClassify;
			break;
		case 1:
			classifier = iftFeatSelSVMRBFClassify;
			break;
		default:
			iftError("Invalid classifier.", "main");
			return 1;
	}

	switch(metricOption)
	{
		case 0:
			metric = iftSkewedTruePositives;
			break;
		case 1:
			metric = iftTruePositives;
			break;
		case 2:
			metric = iftClassifAccuracy;
			break;
		default:
			iftError("Invalid metric.", "main");
			return 1;
	}	
	
	splitDataSet(Z, &Zlearn, &Zeval, &Ztest, z1Perc+z2Perc, z3Perc, z4Perc);
	
	featSelector = iftCreateFeatureSelector(Zlearn, metric, classifier);
	featSelector->featsPenalty = featsPenalty;
	featSelector->nruns = nruns;
	featSelector->trainPerc = z1Perc/(z1Perc+z2Perc);

	classifier(Ztest, featSelector);
	
	printf("Raw Acc = %f\n", iftClassifAccuracy(Ztest));

	sumRank = iftAllocFloatArray(n);
	initFloatArray(sumRank, 0.0, n);

	for (i = 0; i < bootstraps; ++i)
	{
		printf("Bootstrap #%d ...\n", i+1);
		iftFeatureWeighting(featSelector);
		sumFloatArray(sumRank, featSelector->Z->alpha, n);
		iftCopyFloatArray(Z->alpha, featSelector->Z->alpha, n);
	}

	float max = FLT_MAX, threshold, threshold_star, acc_star;
	int counter = 0;
	float* thresholds = iftAllocFloatArray(n);
	iftCopyFloatArray(thresholds, sumRank, n);
	qsort (thresholds, n, sizeof(float), fcompare);

	threshold = threshold_star = acc_star = 0.0f;
	for(i=0; i<n; ++i)
	{
		if((thresholds[i]-threshold) < IFT_EPSILON)
			continue;
		
		threshold = thresholds[i];
		for(j=0;j<n;j++)
		{
			if(sumRank[j]>=threshold)
				Zeval->alpha[j] = sumRank[j];
			else
				Zeval->alpha[j] = 0.0f;
		}

		classifier(Zeval, featSelector);
		float acc = iftClassifAccuracy(Zeval);
		printf("Threshold = %f => Acc = %f\n", threshold, acc);
		
		if(acc>acc_star)
		{
			acc_star = acc;
			threshold_star = threshold;
		}
	}

	for (i = 0; i < n; ++i)
	{
		printf("%f ", sumRank[i]);
	}
	printf("\n");

	printf("Threshold = %f\n", threshold_star);
	printf("Best feature set: ");
	for (i = 0; i < n; ++i)
	{
		if(sumRank[i]<threshold_star)
		{
			sumRank[i]=0.0f;
			counter++;
		}
		printf("%f ", sumRank[i]);
	}
	printf("\n");

	printf("Reduction of: %f%%\n", 100.0*((float)(counter)/n));

	iftCopyFloatArray(Ztest->alpha, sumRank, n);
	classifier(Ztest, featSelector);

	printf("Reducted Acc = %f\n", iftClassifAccuracy(Ztest));
	free(sumRank);

	return 0;
}
