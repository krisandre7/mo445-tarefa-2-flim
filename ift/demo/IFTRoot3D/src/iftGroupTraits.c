#include "iftRoot3D.h"

/* 
   Author: Alexandre Falcao.  

   Description: This program reads a phenotype file and the
   SelectedTraits.csv file and computes groups by optimum-path
   forest. The resulting groups are updated in the phenotype file.

*/


int main(int argc, char *argv[]) 
{
  timer          *t1=NULL,*t2=NULL;
  iftPheno       *pheno=NULL,*sortedpheno=NULL;
  iftDataSet     *Z=NULL,*Zaux=NULL;
  iftKnnGraph    *graph=NULL;
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
    iftError("Usage: iftGroupTraits  <phenotype file> <maximum scale (0,1]>","main");

  t1     = iftTic();
  
  pheno = iftReadPhenotypeInfo(argv[1]);
  
  Zaux   = iftPhenoToDataset(pheno);
  iftSetStatus(Zaux,IFT_TRAIN);
  Z      = iftNormalizeDataSet(Zaux);
  iftDestroyDataSet(&Zaux);

  iftReadSelectedTraits("SelectedTraits.csv",Z); 
  kmax   = (int)(atof(argv[2])*Z->nsamples); 
  if (kmax <= 0)  
     iftError("Scale is too low","main"); 

  fprintf(stdout,"Grouping with kmax=%d.\n",kmax); 

  iftSelectUnsupTrainSamples(Z,1.0);
  graph = iftCreateKnnGraph(Z,kmax);
  iftUnsupTrain(graph,iftNormalizedCut);
  iftUnsupClassify(graph,Z);
  iftCopyIndivLabels(Z,pheno); 
  sortedpheno = iftSortIndivByLabel(pheno); 
  iftWritePhenotypeInfo(sortedpheno,argv[1]); 
  iftCompNumIndivsPerGroup(sortedpheno);

  t2     = iftToc();
  fprintf(stdout,"clustering in %f ms\n",iftCompTime(t1,t2));
  
  iftDestroyPheno(&pheno);
  iftDestroyDataSet(&Z); 
  iftDestroyKnnGraph(&graph); 
  iftDestroyPheno(&sortedpheno); 

  /* ---------------------------------------------------------- */
  

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



