#include "ift.h"
#include <ctype.h>
#include <malloc.h>

#define _SILENCE

enum { DEMO_LINEAR, DEMO_RBF, DEMO_PRECOMPUTED }; /* demo kernel enumeration */


int main(int argc, char *argv[]) 
{
  iftDataSet      *Z_train = NULL, *Z1[3];
  iftSVM          *svm=NULL;
  timer           *t1=NULL,*t2=NULL;
  int              i, num_of_comps, reduction;

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

  if (argc != 6)
    iftError("Usage: iftClassifyBySVM <train_dataset.dat>" \
             " <preprocess_type [0=NoReduction, 1=PCA, 2=SupPCA]>" \
             " <num_of_comps>" \
             " <kernel_type [0=linear,1=RBF,2=precomputed]" \
             " <multiclass [0=OVO,1=OVA]>",
             "main");

  // <train_dataset.dat> <preprocess_type [0=NoReduction, 1=PCA, 2=SupPCA]>
  // <num_of_comps> <kernel_type [0=linear,1=RBF,2=precomputed]" <multiclass [0=OVO,1=OVA]>"


//  iftRandomSeed();

  /* Initialization */

  Z_train  = iftReadOPFDataSet(argv[1]); // Read trainning dataset Z_train

#ifndef _SILENCE
  printf("***** TRAINING DATASET *****\n");
  printf("Total number of samples  %d\n",Z_train->nsamples);
  printf("Total number of features %d\n",Z_train->nfeats);
  printf("Total number of classes  %d\n\n",Z_train->nclasses);
#endif 

  iftSetDistanceFunction(Z_train, 1);
  iftSetStatus(Z_train, IFT_TRAIN); // set all elements as IFT_TRAIN

  reduction    = atoi(argv[2]);
  num_of_comps = atoi(argv[3]);

  if ((num_of_comps <= 0) && (reduction > 0))
    iftError("Cannot reduce feature space to 0 or less components",
             "iftDataClassifyByKNN");
  
  // SVM
  int kernel_type = atoi(argv[4]);
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
    Z1[i]=NULL;
  }

  if (Z1[2] != NULL) iftDestroyDataSet(&Z1[2]);

  Z1[0] = iftCopyDataSet(Z_train);

  switch(reduction) {
  case 0:
    Z1[2] = iftNormalizeDataSet(Z1[0]);
    iftDestroyDataSet(&Z1[0]);

    // saving feature normalization vector 
    char filem[200],files[200];
    sprintf(filem,"mean.%d.data",Z1[2]->nfeats);
    sprintf(files,"stdev.%d.data",Z1[2]->nfeats);
    FILE* pFm = fopen(filem,"wb+");      
    FILE* pFs = fopen(files,"wb+");      
    if ( (pFm == NULL) || (pFs == NULL) ) 
      iftError("Can't create mean feature file","iftMobILiveModel");
    fwrite(Z1[2]->fsp.mean ,sizeof(float),Z1[2]->nfeats,pFm);
    fwrite(Z1[2]->fsp.stdev,sizeof(float),Z1[2]->nfeats,pFs);
    fclose(pFm);fclose(pFs);
    break;
  case 1:
    Z1[1] = iftNormalizeDataSet(Z1[0]);
    iftDestroyDataSet(&Z1[0]);
    Z1[2] = iftTransFeatSpaceByPCA(Z1[1],num_of_comps);
    iftDestroyDataSet(&Z1[1]);
    // ToDo's
    break;
  case 2:
    Z1[1] = iftNormalizeDataSet(Z1[0]);
    iftDestroyDataSet(&Z1[0]);
    Z1[2] = iftTransFeatSpaceBySupPCA(Z1[1],num_of_comps);
    iftDestroyDataSet(&Z1[1]);
    // ToDo's
    break;
  default:
    iftError("Invalid reduction option","iftDataClassifyBySVM");
  }


  if (kernel_type == DEMO_PRECOMPUTED) {
    iftDataSet *Zaux, *Z1k, *Z2k;
    uchar traceNormalize = 1;
    float ktrace;

    Z1k = iftKernelizeDataSet2(Z1[2], Z1[2], LINEAR, traceNormalize, &ktrace);

    if (ktrace != 0.0)
      iftMultDataSetByScalar(Z2k, 1.0 / ktrace);

    Zaux = Z1[2];
    Z1[2] = Z1k;
    iftDestroyDataSet(&Zaux);

    // ToDo's
  }

  if (atoi(argv[5])==0){
    iftSVMTrainOVO(svm, Z1[2]); // Training
    // ToDo's
  }else{
    iftSVMTrainOVA(svm, Z1[2]);
    if (Z1[2]->nclasses == 2) {
      float rho;
      iftSample w = iftSVMGetNormalHyperplane(svm,0,Z1[2],&rho);
      char fileHyper[200];
      char fileRho[200];
      sprintf(fileHyper,"hyperplane.%d.data",Z1[2]->nfeats);
      sprintf(fileRho,"rho.%d.data",Z1[2]->nfeats);
      FILE *pFH = fopen(fileHyper,"wb+");
      FILE *pFR = fopen(fileRho  ,"wb+");
      if ( (pFH == NULL) || (pFR == NULL) ) 
	iftError("Can't save hyperplane and details files","iftMobILiveModel");
      fwrite((w.feat),sizeof(float),Z1[2]->nfeats,pFH);
      fprintf(stdout,"rho: %f, class: %lf %lf\n",rho,svm->class[0],svm->class[1]);
      fwrite(&rho,sizeof(float),1,pFR);
      fwrite(&(svm->class[0]),sizeof(double),1,pFR);
      fwrite(&(svm->class[1]),sizeof(double),1,pFR);
      fclose(pFH);
      fclose(pFR);
    }
  }

  iftDestroyDataSet(&Z1[2]);

  iftDestroySVM(svm);
  iftDestroyDataSet(&Z_train);

  t2     = iftToc();

#ifndef _SILENCE
  fprintf(stdout,"analysis  in %f ms\n",iftCompTime(t1,t2));
#endif
 

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





