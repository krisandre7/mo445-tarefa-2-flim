#include "ift.h"
#include <ctype.h>
#include <malloc.h>

#define _SILENCE

enum { DEMO_LINEAR, DEMO_RBF, DEMO_PRECOMPUTED }; /* demo kernel enumeration */


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

void iftFalseRates(iftDataSet *Z,float* pfar,float* pfrr,float* pffs){
  int i, ntruefake=0, nfalsefake=0, nfalsereal=0, ntruereal=0;
  // 1 - fake,  2 - real
  for (i = 0; i < Z->nsamples; i++){
    if (Z->sample[i].status != IFT_TRAIN){
      if ((Z->sample[i].class == 1) && (Z->sample[i].label == 1)) ntruefake++;
      if ((Z->sample[i].class == 1) && (Z->sample[i].label == 2)) nfalsereal++;
      if ((Z->sample[i].class == 2) && (Z->sample[i].label == 1)) nfalsefake++;
      if ((Z->sample[i].class == 2) && (Z->sample[i].label == 2)) ntruereal++;
    }
  }
  
  *pfar = (float)nfalsereal /(float)(nfalsereal+ntruefake);
  *pfrr = (float)nfalsefake /(float)(nfalsefake+ntruereal);
  *pffs = 2*(1-*pfar)*(1-*pfrr)/((1-*pfar)+(1-*pfrr));

  //  fprintf(stdout,"tr:%3d,tf:%3d,fr:%3d,ff:%3d -",ntruereal,ntruefake,nfalsereal,nfalsefake);

}

int main(int argc, char *argv[]) 
{
  iftDataSet      *Z_train = NULL, *Z_test = NULL, *Z1[3], *Z2[3];
  iftSVM          *svm=NULL;
  timer           *t1=NULL,*t2=NULL;
  float            stdev, mean, *acc, *far, *frr, *ffs;
  float            perc_train;
  int              i, n, num_of_comps, reduction;

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
    iftError("Usage: iftClassifyBySVM <train_dataset.dat> <test_dataset.dat|NULL> <perc_train> <niters>" \
             " <preprocess_type [0=NoReduction, 1=PCA, 2=SupPCA]>" \
             " <num_of_comps>" \
             " <kernel_type [0=linear,1=RBF,2=precomputed]" \
             " <multiclass [0=OVO,1=OVA]>",
             "main");

  // <train_dataset.dat> <test_dataset.dat> <niters> <preprocess_type [0=NoReduction, 1=PCA, 2=SupPCA]>
  // <num_of_comps> <kernel_type [0=linear,1=RBF,2=precomputed]" <multiclass [0=OVO,1=OVA]>"


//  iftRandomSeed();

  /* Initialization */

  Z_train  = iftReadOPFDataSet(argv[1]); // Read trainning dataset Z_train
  if (strcmp(argv[2],"NULL") != 0)
    Z_test  = iftReadOPFDataSet(argv[2]); // Read testing Z_test
  else 
    Z_test = NULL;

  n    = atoi(argv[4]);
  acc  = iftAllocFloatArray(n); 
  if (Z_train->nclasses == 2) {
    far = iftAllocFloatArray(n); 
    frr = iftAllocFloatArray(n); 
    ffs = iftAllocFloatArray(n); 
  }

#ifndef _SILENCE
  printf("***** TRAINING DATASET *****\n");
  printf("Total number of samples  %d\n",Z_train->nsamples);
  printf("Total number of features %d\n",Z_train->nfeats);
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
  perc_train = atof(argv[3]);

  num_of_comps = atoi(argv[6]);
  reduction    = atoi(argv[5]);

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
    //    usleep(5);
    if (Z1[2] != NULL) iftDestroyDataSet(&Z1[2]);
    if (Z2[2] != NULL) iftDestroyDataSet(&Z2[2]);

    if (Z_test == NULL) {
      iftSetStatus(Z_train, IFT_TRAIN); // set all elements as IFT_TRAIN
      iftSelectSupTrainSamples(Z_train,perc_train); // Select training samples
      Z1[0] = iftExtractSamples(Z_train,IFT_TRAIN);
      Z2[0] = iftExtractSamples(Z_train,IFT_TEST);
    } else {
      Z1[0] = iftCopyDataSet(Z_train);
      Z2[0] = iftCopyDataSet(Z_test);
    }

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

      Z1k = iftKernelizeDataSet2(Z1[2], Z1[2], LINEAR, traceNormalize, &ktrace);
      Z2k = iftKernelizeDataSet2(Z1[2], Z2[2], LINEAR, 0, &ktrace);

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
        iftSVMLinearClassifyOVA(svm, Z2[2], Z1[2], IFT_TEST); // Classification
      else
        iftSVMClassifyOVA(svm, Z2[2], IFT_TEST); // Classification
//      iftSVMClassifyOVA(svm, Z2[2], IFT_TEST); // Classification
    }

//    acc[i] = iftSkewedTruePositives(Z2[2]); // Compute accuracy on test set
    if (Z2[2]->nclasses == 2) {
	acc[i] = iftTruePositives(Z2[2]); // Compute accuracy on test set
	iftFalseRates(Z2[2],&(far[i]),&(frr[i]),&(ffs[i]));
    } else {
	acc[i] = iftTruePositives(Z2[2]); // Compute accuracy on test set
    }

#ifndef _SILENCE
    if (Z2[2].nclasses == 2) {
      printf("acc[%d] = %f, far[%d] = %f, frr[%d] = %f, ffs[%d] = %f\n"
	     ,i,acc[i],i,far[i],i,frr[r],i,ffs[i]);
    } else {
      printf("acc[%d] = %f\n",i,acc[i]);
    }

#endif

    if (i != n-1) {
      iftDestroyDataSet(&Z1[2]);
      iftDestroyDataSet(&Z2[2]);
    }
  }

  if (atoi(argv[8])==0)
    iftSVMClassifyOVO(svm, Z2[2], IFT_TRAIN); // just to complete the label map
  else
    iftSVMClassifyOVA(svm, Z2[2], IFT_TRAIN);

  iftDestroySVM(svm);
  iftDestroyDataSet(&Z_train);
  if (Z_test != NULL)
    iftDestroyDataSet(&Z_test);

  t2     = iftToc();

#ifndef _SILENCE
  fprintf(stdout,"classifications in %f ms\n",iftCompTime(t1,t2));
#endif
 

  if (Z2[2]->nclasses == 2) {
    float meanfar=0.,meanfrr=0.,meanffs=0.,stdevfar=0.,stdevfrr=0.,stdevffs=0.;
    iftMeanStd(acc,n,&mean,&stdev);
    iftMeanStd(far,n,&meanfar,&stdevfar);
    iftMeanStd(frr,n,&meanfrr,&stdevfrr);
    iftMeanStd(ffs,n,&meanffs,&stdevffs);
    fprintf(stdout,"acc=%7.3f(%6.4f), far=%7.3f(%6.4f), frr=%7.3f(%6.4f), ffs=%7.3f(%6.4f)\n",
	    100.*mean,100.*stdev,100.*meanfar,100.*stdevfar,100.*meanfrr,100.*stdevfrr,100.*meanffs,100.*stdevffs);
    free(far);free(frr);free(ffs);
  } else {
    iftMeanStd(acc,n,&mean,&stdev);
    fprintf(stdout,"Accuracy of classification is mean=%7.3f, stdev=%7.4f\n",100.*mean,100.*stdev); 
  }
  free(acc);

  if (Z2[2]->nfeats == 2){
    iftImage *img = iftDraw2DFeatureSpace(Z2[2],LABEL,0);
    iftWriteImageP6(img,"labels.ppm");
    iftDestroyImage(&img);
    img = iftDraw2DFeatureSpace(Z2[2],CLASS,0);
    iftWriteImageP6(img,"classes.ppm");
    iftDestroyImage(&img);
    img = iftDraw2DFeatureSpace(Z1[2],STATUS,IFT_TRAIN);
    iftWriteImageP6(img,"train.ppm");
    iftDestroyImage(&img);
  }

  iftDestroyDataSet(&Z1[2]);
  iftDestroyDataSet(&Z2[2]);
  
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





