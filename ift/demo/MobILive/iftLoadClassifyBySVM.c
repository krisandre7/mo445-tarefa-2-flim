#include "ift.h"
#include <ctype.h>
#include <malloc.h>

#define _SILENCE

enum { DEMO_LINEAR, DEMO_RBF, DEMO_PRECOMPUTED }; /* demo kernel enumeration */

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
  
  *pfar = (float)nfalsereal /(nfalsereal+ntruefake);
  *pfrr = (float)nfalsefake /(nfalsefake+ntruereal);
  *pffs = 2*(1-*pfar)*(1-*pfrr)/((1-*pfar)+(1-*pfrr));
}

int main(int argc, char *argv[]) 
{
  iftDataSet      *Z_test = NULL;
  timer           *t1=NULL,*t2=NULL;
  int              i,nimages;

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

  if (argc != 4)
    iftError("Usage: iftClassifyBySVM <test_dataset.dat> <norm_features.data> <model.data>", 
             "iftLoadClassifyBySVM");

  // iftRandomSeed();

  Z_test  = iftReadOPFDataSet(argv[1]); // Read testing Z_test


  iftSetDistanceFunction(Z_test, 1);
#ifndef _SILENCE
  printf("***** TESTING DATASET *****\n");
  printf("Total number of samples  %d\n",Z_test->nsamples);
  printf("Total number of features %d\n",Z_test->nfeats);
  printf("Total number of classes  %d\n\n",Z_test->nclasses);
#endif

  iftSetDistanceFunction(Z_test, 1);
  iftSetStatus(Z_test, IFT_TEST); // set all elements as IFT_TEST


  // Loading feature normalization files
  float* mean  = iftAllocFloatArray(Z_test->nfeats);
  float* stdev = iftAllocFloatArray(Z_test->nfeats);

  char filen[200];
  strcpy(filen,argv[2]);
  FILE* pFn = fopen(filen,"rb");      
  if ( (pFn == NULL) ) 
    iftError("Can't open feature normalization file","LIV.IC.UNICAMP.MobILiveVerify");
  if ( fread(mean ,sizeof(float),Z_test->nfeats,pFn) != Z_test->nfeats)
    iftError("Can't read mean normalization vector","LIV.IC.UNICAMP.MobILiveVerify");
  if ( fread(stdev,sizeof(float),Z_test->nfeats,pFn) != Z_test->nfeats)
    iftError("Can't read stdev normalization vector","LIV.IC.UNICAMP.MobILiveVerify");
  fclose(pFn);


  // Apply classification
  float rho;
  double class[2];
  iftSample w;
  w.feat = (float*) iftAllocFloatArray(Z_test->nfeats);

  char fileModel[200];
  strcpy(fileModel,argv[3]);
  FILE *pFM = fopen(fileModel,"rb");
  if ( (pFM == NULL) ) 
    iftError("Can't read model classification file","LIV.IC.UNICAMP.MobILiveVerify");
  if ( fread((w.feat),sizeof(float),Z_test->nfeats,pFM) != Z_test->nfeats)
    iftError("Can't read hyperplane values","LIV.IC.UNICAMP.MobILiveVerify");
  if ( fread(&rho,sizeof(float),1,pFM) != 1)
    iftError("Can't read bias hyperplane value","LIV.IC.UNICAMP.MobILiveVerify");
  if ( fread(&(class[0]),sizeof(double),1,pFM) != 1)
    iftError("Can't read class FAKE information","LIV.IC.UNICAMP.MobILiveVerify");
  if ( fread(&(class[1]),sizeof(double),1,pFM) != 1)
    iftError("Can't red class REAL information","LIV.IC.UNICAMP.MobILiveVerify");
  //  fprintf(stdout,"rho: %f, cl0: %lf, cl1: %lf\n",rho,svm->class[0],svm->class[1]);
  fclose(pFM);

  t1     = iftTic();

  nimages = Z_test->nsamples;
  for (int i = 0; i <nimages ; i++) {
    // Feature normalization
    for ( int f=0; f < Z_test->nfeats; f++) { 
      Z_test->sample[i].feat[f] = Z_test->sample[i].feat[f]-mean[f];
      if (stdev[f]> Epsilon)
	Z_test->sample[i].feat[f] /= stdev[f];
    }

    // Sample Classification
    int imaxpred;
    float prediction[2];
    prediction[0] = 0.;
    for( int f = 0; f<Z_test->nfeats; f++) 
      prediction[0] += w.feat[f] * Z_test->sample[i].feat[f];
    prediction[0] -= rho;

    prediction[1] = -prediction[0];

    if (prediction[0] > prediction[1] )
      imaxpred = 0;
    else
      imaxpred = 1;

    Z_test->sample[i].label  = class[imaxpred];
    Z_test->sample[i].weight = prediction[imaxpred];
    //fprintf(stdout,"%f %f %6.4f\t",prediction[0],prediction[1],Z_test->sample[i].weight);

    //    //    define label
    //    if (Z_test->sample[i].label == 1) {
    //      fprintf(stdout,"%d\tFake\n",i);
    //    } else { // (label == 2)
    //      fprintf(stdout,"%d\tReal\n",i);
    //    }
  }

  free(w.feat);

  float acc=0.,far=0.,frr=0.,ffs=0.;
  acc = iftTruePositives(Z_test); // Compute accuracy on test set
  iftFalseRates(Z_test,&far,&frr,&ffs);


  iftDestroyDataSet(&Z_test);

  t2     = iftToc();

#ifndef _SILENCE
  fprintf(stdout,"classifications in %f ms\n",iftCompTime(t1,t2));
#endif

  fprintf(stdout,"acc=%.4f, far=%.4f, frr=%.4f, ffs=%.4f\n",
	  acc,far,frr,ffs);

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





