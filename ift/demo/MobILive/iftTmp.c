#include "ift.h"
#include <ctype.h>
#include <malloc.h>

//#define _SILENCE
#define _VERBOSE

enum { DEMO_LINEAR, DEMO_RBF, DEMO_PRECOMPUTED }; /* demo kernel enumeration */

// Considering always:
#define NIMGSB   8

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
  iftDataSet      *Z_train = NULL,*Z_test = NULL, *Z1[3], *Z2[3];
  iftSVM          *svm=NULL;
  timer           *t1=NULL,*t2=NULL;
  float            stdev, mean, *acc,accmax,accmin, *far, *frr, *ffs;
  float            perc_train,ntrainsubj;
  int              it, i, j, n, sb, s, id, high, t, num_of_comps, reduction, status;
  int             *sample,*count,**msubj,*houtliers,*houtsort;
  int              NSUBJ,NIMGCL;

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

  if (argc != 10)
    iftError("Usage: iftSubjClassifyBySVM <train_dataset.dat> <NULL|test_dataset.dat> <nsubj> <perc_train> <niters>" \
             " <preprocess_type [0=NoReduction, 1=PCA, 2=SupPCA]>" \
             " <num_of_comps>" \
             " <kernel_type [0=linear,1=RBF,2=precomputed]" \
             " <multiclass [0=OVO,1=OVA]>",
             "main");

  NSUBJ        = atoi(argv[3]);
  NIMGCL       = NSUBJ * NIMGSB;
  perc_train   = atof(argv[4]);
  n            = atoi(argv[5]);
  reduction    = atoi(argv[6]);
  num_of_comps = atoi(argv[7]);

  ntrainsubj = (int)(perc_train*NSUBJ+0.5);

  iftRandomSeed(1);

  // Allocating subjects per partition
  msubj = (int**)malloc(sizeof(int*)*n);
  msubj[0] = iftAllocIntArray(n*NSUBJ);
  for(i=1;i<n;i++) msubj[i]=msubj[0]+i*NSUBJ;
  sample    = iftAllocIntArray(NSUBJ); 
  count     = iftAllocIntArray(NSUBJ); 
  houtliers = iftAllocIntArray(NIMGCL*2);
  houtsort  = iftAllocIntArray(NIMGCL*2);

  // Randomly select samples

  for(it=0;it<n;it++) {
    // Prepare samples for selection
    t=0;
    for (s=0; s < NSUBJ; s++) {
      sample[t]=s;
      count[t]=100;
      t++;
    }
    t = 0; high = NSUBJ-1;
    while (t < ntrainsubj) {
      i = iftRandomInteger(0,high);
      s = sample[i];
      if (count[i]==0){
	msubj[it][s]=1; // for training
	iftSwitchValues(&sample[i],&sample[high]);
	iftSwitchValues(&count[i],&count[high]);
	t++; high--;
      }else{
	count[i]--;
      }
    }
    // for(t=0;t<NSUBJ;t++)
    //   fprintf(stdout,"%1d ",msubj[it][t]);
    // fprintf(stdout,"\n");
  }
  free(count);
  free(sample);


//  fprintf(stdout,"train16:");
//  for(i=0;i<NSUBJ;i++)
//    if (msubj[3][i]==1)
//	fprintf(stdout," %02d",i);
//  fprintf(stdout,"\n");

//  fprintf(stdout,"train16:");
//  for(i=0;i<NSUBJ;i++)
//    if (msubj[16][i]==1)
//	fprintf(stdout," %02d",i);
//  fprintf(stdout,"\n");
	
//  fprintf(stdout,"train69:");
//  for(i=0;i<NSUBJ;i++)
//    if (msubj[69][i]==1)
//      fprintf(stdout," %02d",i);
//  fprintf(stdout,"\n");

  /* Initialization */
  Z_train = iftReadOPFDataSet(argv[1]); // Read dataset Z
  if (strcmp(argv[2],"NULL") != 0)
    Z_test  = iftReadOPFDataSet(argv[2]); // Read testing Z_test
  else 
    Z_test = NULL;
  if ( (Z_train->nsamples != NIMGCL*2) || (Z_test != NULL && Z_test->nsamples != NIMGCL*2) ){
    char msg[200];
    sprintf(msg,"The number of samples in datasets is different from required\n",Z_train->nsamples,NIMGCL*2);
    iftError(msg,"iftSubjClassifyBySVM");
  }

  acc = iftAllocFloatArray(n);   accmax = 0.; accmin = 0.;
  far = iftAllocFloatArray(n); 
  frr = iftAllocFloatArray(n); 
  ffs = iftAllocFloatArray(n); 

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
  int kernel_type = atoi(argv[8]);
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
    for(sb=0;sb<NSUBJ;sb++) {
      if ( msubj[i][sb] == 1 ) {
	status = IFT_TRAIN;
      } else {
	status = IFT_TEST;
      }

      for(s=sb*NIMGSB;s<=(sb+1)*NIMGSB-1;s++) {
	Z_train->sample[s             ].status = status; // cl1
	Z_train->sample[s+NIMGSB*NSUBJ].status = status; // cl2
	if (Z_test != NULL) {
	  Z_test->sample[s             ].status = status; // cl1
	  Z_test->sample[s+NIMGSB*NSUBJ].status = status; // cl2
	}
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

    if (atoi(argv[9])==0){
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
    if ( (i == 0) || (acc[i] > accmax) ) accmax = acc[i];
    if ( (i == 0) || (acc[i] < accmin) ) accmin = acc[i];

#ifndef _SILENCE
    if (Z2[2]->nclasses == 2) {
      printf("%3d: acc = %7.3f, far = %7.3f, frr = %7.3f, ffs = %7.3f\n"
	     ,i,100.*acc[i],100.*far[i],100.*frr[i],100.*ffs[i]);
    } else {
      printf("acc[%d] = %f\n",i,acc[i]);
    }
#endif
    if (acc[i] != 1.) {
      for(s=0;s<Z2[2]->nsamples;s++)
	if (Z2[2]->sample[s].class != Z2[2]->sample[s].label) {
	  id = Z2[2]->sample[s].id;
	  houtliers[id]++;
#ifndef _SILENCE
	  fprintf(stdout,"%06d-%06d, subject: %03d-%d\n"
             ,id/(NIMGCL)+1,id%(NIMGCL)+1
             ,(id%NIMGCL)/NIMGSB+1,(id%NIMGCL)%NIMGSB+1);
#endif
	}
    }

    if (i != n-1) {
      iftDestroyDataSet(&Z1[2]);
      iftDestroyDataSet(&Z2[2]);
    }
  }

  iftDestroySVM(svm);
  iftDestroyDataSet(&Z_train);
  if (Z_test != NULL)
    iftDestroyDataSet(&Z_test);

  // sorting houtliers[s]
  for(i=0;i<NIMGCL*2-1;i++) {
    for(t=i,j=i+1;j<NIMGCL*2;j++) {
      if (houtliers[t] < houtliers[j])
	t = j;
    }
    if (houtliers[i] != 0 || houtliers[t] != 0) {
      // fprintf(stdout,"values: %d(%d)/%d(%d)\n",houtliers[i],i,houtliers[t],t);
      int aux = houtliers[i];
      houtliers[i] = houtliers[t];
      houtliers[t] = aux;
      // fprintf(stdout,"values: %d(%d)/%d(%d)\n",houtliers[i],i,houtliers[t],t);
    }
    houtsort[i] = t;
  }


#ifdef _VERBOSE
  fprintf(stdout,"outliers summary\n");
#endif
  int totoutliers=0,toterrors=0;
  for(s=0;s<NIMGCL*2;s++)
    if (houtliers[s] != 0) {
      id = houtsort[s];
#ifdef _VERBOSE
      fprintf(stdout,"%06d-%06d, subject: %03d-%d, %3d\n"
	      ,id/(NIMGCL)+1,id%(NIMGCL)+1
	      ,(id%NIMGCL)/NIMGSB+1,(id%NIMGCL)%NIMGSB+1
	      ,houtliers[s]);
#endif
      totoutliers++;
      toterrors+=houtliers[s];
    }
#ifdef _VERBOSE
  fprintf(stdout,"number of 'outliers': %d / total errors: %d\n",totoutliers,toterrors);
#endif
  free(houtliers);
  free(houtsort);

  t2     = iftToc();

#ifndef _SILENCE
  fprintf(stdout,"classifications in %f ms\n",iftCompTime(t1,t2));
#endif
 

  float meanfar=0.,meanfrr=0.,meanffs=0.,stdevfar=0.,stdevfrr=0.,stdevffs=0.;
  iftMeanStd(acc,n,&mean,&stdev);
  iftMeanStd(far,n,&meanfar,&stdevfar);
  iftMeanStd(frr,n,&meanfrr,&stdevfrr);
  iftMeanStd(ffs,n,&meanffs,&stdevffs);
  fprintf(stdout,"acc=%7.3f(%6.4f)[%7.3f,%7.3f], far=%7.3f(%6.4f), frr=%7.3f(%6.4f), ffs=%7.3f(%6.4f)\n",
	  100.*mean,100.*stdev,100.*accmin,100.*accmax,
	  100.*meanfar,100.*stdevfar,100.*meanfrr,100.*stdevfrr,100.*meanffs,100.*stdevffs);
  free(far);free(frr);free(ffs);free(acc);

  free(msubj[0]);free(msubj);

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
