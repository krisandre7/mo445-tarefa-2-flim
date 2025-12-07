#include "iftRoot3D.h"

/* 
   Author: Alexandre Falcao.  

   Description: This program reads the Phenotype_ALL.csv file and
   selects a given subpopulation, writing a new phenotype file for
   that subpopulation.

*/


int main(int argc, char *argv[]) 
{
  timer          *t1=NULL,*t2=NULL;
  iftPheno       *pheno=NULL;
  iftPheno       *newpheno=NULL;
  char            filename[100];

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=2)
    iftError("Usage: iftSelectSubpop <subpop>","main");

  if (strcmp(argv[1],"ALL")==0){
    iftWarning("No specific subpopulation was selected","main");
    return(0);
  }
  
  t1     = iftTic();

  pheno    = iftReadPhenotypeInfo("Phenotype_ALL.csv");
  newpheno = iftSelectSubpop(pheno,argv[1]);
  sprintf(filename,"Phenotype_%s.csv",argv[1]);
  iftWritePhenotypeInfo(newpheno,filename);

  t2     = iftToc();
  fprintf(stdout,"Subpopulation selected in %f ms\n",iftCompTime(t1,t2));

  iftDestroyPheno(&pheno);
  iftDestroyPheno(&newpheno);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



