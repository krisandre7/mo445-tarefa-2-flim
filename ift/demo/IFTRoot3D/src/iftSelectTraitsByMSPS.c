#include "iftRoot3D.h"

/* 
   Author: Alexandre Falcao.  

   Description: This program reads a phenotype file and assigns
   weights to the traits by executing the Multi-Scale Parameter Search
   (MSPS) algorithm. It outputs a trait selection file
   (SelectedTraits.csv) with the weights of each trait.

*/


int main(int argc, char *argv[]) 
{
  timer          *t1=NULL,*t2=NULL;
  iftPheno       *pheno=NULL;
  iftDataSet     *Z=NULL,*Zaux=NULL;
  int             kmax;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=3)
    iftError("Usage: iftSelectTraitsByMSPS <phenotype file> <maximum scale (0,1]>","main");

  iftRandomSeed(IFT_RANDOM_SEED);
  
  t1     = iftTic();
  
  pheno  = iftReadPhenotypeInfo(argv[1]);
  Zaux   = iftPhenoToDataset(pheno);
  iftSetStatus(Zaux,IFT_TRAIN);
  Z      = iftNormalizeDataSet(Zaux);
  iftDestroyDataSet(&Zaux);
  kmax   = (int)(atof(argv[2])*Z->nsamples);
  if (kmax <= 0) 
    iftError("Scale is too low","main");

  fprintf(stdout,"Weighting traits using kmax=%d\n It might take a few seconds.\n",kmax);
  iftUnsupFeatSelecByMSPS(Z,kmax);
  iftWriteSelectedTraits(pheno,Z->alpha,"SelectedTraits.csv");

  t2     = iftToc();
  fprintf(stdout,"Trait selection by VPS in %f ms\n",iftCompTime(t1,t2));

  iftDestroyPheno(&pheno);
  iftDestroyDataSet(&Z);


  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



