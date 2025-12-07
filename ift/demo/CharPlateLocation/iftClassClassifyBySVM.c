#include "ift.h"
#include <ctype.h>
#include <malloc.h>

//#define _SILENCE
#define _VERBOSE

enum { DEMO_LINEAR, DEMO_RBF, DEMO_PRECOMPUTED }; /* demo kernel enumeration */

void iftMeanStd(float* vector,int n,float *pmean,float* pstdev);

int main(int argc, char *argv[]) 
{
  iftDataSet      *Z_train = NULL,*Z_test = NULL, *Z1[3], *Z2[3];
  iftSVM          *svm=NULL;
  timer           *t1=NULL,*t2=NULL;
  float            stdev, mean, *acc,accmax,accmin;
  float            perc_train;
  int              ntrain,ntraincl;
  int              it, i, j, n, s, id, c, high, t, num_of_comps, reduction, status;
  int             *sample,*count,**mcl,*ncl;

  /*--------------------------------------------------------*/

#ifndef _SILENCE
  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif

  /*--------------------------------------------------------*/

  if (argc != 9)
    iftError("Usage: iftClassClassifyBySVM <train_dataset.dat> <NULL|test_dataset.dat> <perc_train> <niters>" \
             " <preprocess_type [0=NoReduction, 1=PCA, 2=SupPCA]>" \
             " <num_of_comps>" \
             " <kernel_type [0=linear,1=RBF,2=precomputed]" \
             " <multiclass [0=OVO,1=OVA]>",
             "main");

  /* Initialization */
  Z_train = iftReadOPFDataSet(argv[1]); // Read dataset Z
  if (strcmp(argv[2],"NULL") != 0)
    Z_test  = iftReadOPFDataSet(argv[2]); // Read testing Z_test
  else 
    Z_test = NULL;
//  if ( (Z_train->nsamples != NIMGCL*2) || (Z_test != NULL && Z_test->nsamples != NIMGCL*2) ){
//    char msg[200];
//    sprintf(msg,"The number of samples in datasets is different from required\n",Z_train->nsamples,NIMGCL*2);
//    iftError(msg,"iftSubjClassifyBySVM");
//  }

  perc_train   = atof(argv[3]);
  n            = atoi(argv[4]);
  reduction    = atoi(argv[5]);
  num_of_comps = atoi(argv[6]);

  ntrain   = (int)(perc_train*Z_train->nsamples+0.5);
  ntraincl = (int)(ntrain/Z_train->nclasses);

  iftRandomSeed(1);

  // Allocating subjects per partition
  mcl     = (int**)malloc(sizeof(int*)*n);
  mcl[0]  = iftAllocIntArray(n*Z_train->nsamples);
  ncl     = iftAllocIntArray(Z_train->nclasses+1);
  for(i=1;i<n;i++) mcl[i]=mcl[0]+i*Z_train->nsamples;
  sample    = iftAllocIntArray(Z_train->nsamples); 
  count     = iftAllocIntArray(Z_train->nsamples); 


  // Randomly select samples

  for(it=0;it<n;it++) {
    // Prepare samples for selection
    t=0;
    for (s=0; s < Z_train->nsamples; s++) {
      sample[t]=s;
      count[t]=100;
      t++;
    }
    for (c=1; c <= Z_train->nclasses; c++) {
      ncl[c]=0;
    }
    t = 0; high = Z_train->nsamples-1;
    while (t < Z_train->nsamples) {
      i = iftRandomInteger(0,high);
      s = sample[i];
      if (count[i]==0){
        id = Z_train->sample[s].truelabel;
        if (ncl[id] < ntraincl) {
          mcl[it][s]=1; // for training
          ncl[id]++;
        }
	iftSwitchValues(&sample[i],&sample[high]);
	iftSwitchValues(&count[i],&count[high]);
	t++; high--;
      }else{
	count[i]--;
      }
    }
    fprintf(stdout,"computed: ");
    for(c=1;c<=Z_train->nclasses;c++) {
      fprintf(stdout,"%02d: %d, ",c,ncl[c]);
      ncl[c] = 0;
    }
    fprintf(stdout,"\n");

    for(s=0;s<Z_train->nsamples;s++) {
      id = Z_train->sample[s].truelabel;
      if (mcl[it][s] == 1)
        ncl[id]++;
    }

    fprintf(stdout,"counted: ");
    for(c=1;c<=Z_train->nclasses;c++) {
      fprintf(stdout,"%02d: %d, ",c,ncl[c]);
    }
    fprintf(stdout,"\n\n");
  }
  free(ncl);
  free(count);
  free(sample);

  acc = iftAllocFloatArray(n);   accmax = 0.; accmin = 0.;

#ifndef _SILENCE
  printf("***** DATASET *****\n");
  printf("Total number of samples  %d\n"  ,Z_train->nsamples);
  printf("Total number of features %d\n"  ,Z_train->nfeats);
  printf("Total number of classes  %d\n\n",Z_train->nclasses);
#endif 

  iftSetDistanceFunction(Z_train, 1);
  iftSetStatus(Z_train, IFT_TRAIN); // set all elements as IFT_TRAIN

  if (Z_test != NULL) {
#ifndef _SILENCE
    printf("***** TESTING DATASET *****\n");
    printf("Total number of samples  %d\n",Z_test->nsamples);
    printf("Total number of features %d\n",Z_test->nfeats);
    printf("Total number of classes  %d\n\n",Z_test->nclasses);
#endif
    iftSetDistanceFunction(Z_test, 1);
    iftSetStatus(Z_test, IFT_TEST); // set all elements as IFT_TEST
  }


  if ((num_of_comps <= 0) && (reduction > 0))
    iftError("Cannot reduce feature space to 0 or less components",
             "iftDataClassifyByKNN");
  
  // SVM
  int kernel_type = atoi(argv[7]);
  float C = 1e5;
  float sigma = 0.1;

  switch(kernel_type){
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
      iftError("Invalid kernel type","main");
  }

  t1     = iftTic();
  
  for (i=0; i < 3; i++){
    Z1[i]=Z2[i]=NULL;
  }

  for (i=0; i < n; i++) {
    if (Z1[2] != NULL) iftDestroyDataSet(&Z1[2]);
    if (Z2[2] != NULL) iftDestroyDataSet(&Z2[2]);


    // iftSetStatus(Z, IFT_TRAIN); // set all elements as IFT_TRAIN
    for(s=0;s<Z_train->nsamples;s++) {
      if ( mcl[i][s] == 1 ) {
	status = IFT_TRAIN;
      } else {
	status = IFT_TEST;
      }

      Z_train->sample[s].status = status; // cl1
      if (Z_test != NULL) {
        Z_test->sample[s].status = status; // cl1
      }
    }

    Z1[0] = iftExtractSamples(Z_train,IFT_TRAIN);
    if (Z_test == NULL) {
      Z2[0] = iftExtractSamples(Z_train,IFT_TEST);
    } else {
      Z2[0] = iftExtractSamples(Z_test ,IFT_TEST);
    }
    // fprintf(stdout,"Z1: %d,Z2 :%d\n",Z1[0]->nsamples,Z2[0]->nsamples);

    switch(reduction) {
    case 0:
      Z1[2] = iftNormalizeDataSet(Z1[0]);
      iftDestroyDataSet(&Z1[0]);
      Z2[2] = iftNormalizeTestDataSet(Z1[2],Z2[0]);
      iftDestroyDataSet(&Z2[0]);
      break;
    case 1:
      Z1[1] = iftNormalizeDataSet(Z1[0]);
      iftDestroyDataSet(&Z1[0]);
      Z1[2] = iftTransFeatSpaceByPCA(Z1[1],num_of_comps);
      iftDestroyDataSet(&Z1[1]);
      Z2[1] = iftNormalizeTestDataSet(Z1[2],Z2[0]);
      iftDestroyDataSet(&Z2[0]);
      Z2[2] = iftTransformTestDataSetByPCA(Z1[2],Z2[1]);
      iftDestroyDataSet(&Z2[1]);
      break;
    case 2:
      Z1[1] = iftNormalizeDataSet(Z1[0]);
      iftDestroyDataSet(&Z1[0]);
      Z1[2] = iftTransFeatSpaceBySupPCA(Z1[1],num_of_comps);
      iftDestroyDataSet(&Z1[1]);
      Z2[1] = iftNormalizeTestDataSet(Z1[2],Z2[0]);
      iftDestroyDataSet(&Z2[0]);
      Z2[2] = iftTransformTestDataSetByPCA(Z1[2],Z2[1]);
      iftDestroyDataSet(&Z2[1]);
      break;
    default:
      iftError("Invalid reduction option","iftDataClassifyBySVM");
    }


    if (kernel_type == DEMO_PRECOMPUTED) {
      iftDataSet *Zaux, *Z1k, *Z2k;
      uchar traceNormalize = 1;
      float ktrace;

      Z1k = iftKernelizeDataSet(Z1[2], Z1[2], LINEAR, traceNormalize, &ktrace);
      Z2k = iftKernelizeDataSet(Z1[2], Z2[2], LINEAR, 0, &ktrace);

      if (ktrace != 0.0)
        iftMultDataSetByScalar(Z2k, 1.0 / ktrace);

      Zaux = Z1[2];
      Z1[2] = Z1k;
      iftDestroyDataSet(&Zaux);

      Zaux = Z2[2];
      Z2[2] = Z2k;
      iftDestroyDataSet(&Zaux);

    }

    if (atoi(argv[8])==0){
      iftSVMTrainOVO(svm, Z1[2]); // Training
      iftSVMClassifyOVO(svm, Z2[2], IFT_TEST); // Classification
    }else{
      iftSVMTrainOVA(svm, Z1[2]);
      ///(kernel_type == DEMO_PRECOMPUTED) || (
      if ( (kernel_type == DEMO_LINEAR) )
        iftSVMLinearClassifyOVA(svm, Z2[2], Z1[2], IFT_TEST, NULL); // Classification
      else
        iftSVMClassifyOVA(svm, Z2[2], IFT_TEST); // Classification
//      iftSVMClassifyOVA(svm, Z2[2], IFT_TEST); // Classification
    }

//    acc[i] = iftSkewedTruePositives(Z2[2]); // Compute accuracy on test set
    acc[i] = iftTruePositives(Z2[2]); // Compute accuracy on test set
  
    if ( (i == 0) || (acc[i] > accmax) ) accmax = acc[i];
    if ( (i == 0) || (acc[i] < accmin) ) accmin = acc[i];

#ifndef _SILENCE
    printf("acc[%d] = %f\n",i,acc[i]);
#endif

    if (i != n-1) {
      iftDestroyDataSet(&Z1[2]);
      iftDestroyDataSet(&Z2[2]);
    }
  }

  iftDestroySVM(svm);
  iftDestroyDataSet(&Z_train);
  if (Z_test != NULL)
    iftDestroyDataSet(&Z_test);

  t2     = iftToc();

#ifndef _SILENCE
  fprintf(stdout,"classifications in %f ms\n",iftCompTime(t1,t2));
#endif
 

  iftMeanStd(acc,n,&mean,&stdev);
  fprintf(stdout,"acc=%7.3f(%6.4f)[%7.3f,%7.3f]\n",
	  100.*mean,100.*stdev,100.*accmin,100.*accmax);
  free(acc);

  free(mcl[0]);free(mcl);

  /* ---------------------------------------------------------- */

#ifndef _SILENCE
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   
#endif

  return(0);
}


void iftMeanStd(float* vector,int n,float *pmean,float* pstdev) {
  int i;
  *pmean = 0.0;
  for (i=0; i < n; i++) {
    *pmean = *pmean + vector[i];
  }
  *pmean = *pmean / n;
  *pstdev = 0.0;

  for (i=0; i < n; i++) {
    *pstdev = *pstdev + (vector[i]-*pmean)*(vector[i]-*pmean);
  }
  if (n > 1)
    *pstdev = sqrtf(*pstdev/(n-1));
}
