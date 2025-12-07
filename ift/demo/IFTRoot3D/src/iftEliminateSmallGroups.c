#include "iftRoot3D.h"

/* 
   Author: Alexandre Falcao.  

   Description: This program reads a phenotype file and eliminates
   groups with less than a minimum number of individuals. The
   phenotype file is rewritten with the individuals ordered by group
   label. 

*/

iftPheno *iftEliminateSmallGroups(iftPheno *pheno, int min_nindivs_group);

iftPheno *iftEliminateSmallGroups(iftPheno *pheno, int min_nindivs_group)
{
  iftPheno *reducedpheno=NULL;
  int i,j,l,nindivs,ngroups,*relabel=NULL; 

  /* Compute the new number of individuals from valid groups, the
     number of valid groups, and the new labels for valid groups */

  iftCompNumIndivsPerGroup(pheno);
  if (pheno->ngroups==0) 
    iftError("First, you must create groups","iftEliminateSmallGroups"); 

  relabel = iftAllocIntArray(pheno->ngroups);
  nindivs = 0; ngroups = 0; l = 1;
  for (i=0; i < pheno->ngroups; i++) 
    if (pheno->nindivs_group[i] >= min_nindivs_group){
      nindivs += pheno->nindivs_group[i];
      ngroups ++;
      relabel[i]=l; 
      l++;
    }

  /* Create new phenotype file without the small groups */

  reducedpheno = iftCreatePheno(nindivs,pheno->ntraits); 

  for (j=0; j < pheno->ntraits; j++) 
    strcpy(reducedpheno->traitname[j],pheno->traitname[j]);

  reducedpheno->ngroups       = ngroups;  
  reducedpheno->nindivs_group = iftAllocIntArray(ngroups);
  j = 0; 
  for (i=0; i < pheno->ngroups; i++)
    if (pheno->nindivs_group[i] >= min_nindivs_group){
      reducedpheno->nindivs_group[j] = pheno->nindivs_group[i];
      j++;
    }
  
  l=0; 
  for(i=0; i < pheno->nindivs; i++) {
    if (pheno->nindivs_group[pheno->indiv[i].label-1] >= min_nindivs_group){
      strcpy(reducedpheno->indiv[l].lineID,pheno->indiv[i].lineID);
      strcpy(reducedpheno->indiv[l].subpop,pheno->indiv[i].subpop);
      for (j=0; j < pheno->ntraits; j++) 
	reducedpheno->indiv[l].traitvalue[j] = pheno->indiv[i].traitvalue[j]; 
      reducedpheno->indiv[l].label    = relabel[pheno->indiv[i].label-1];     
      reducedpheno->indiv[l].position = pheno->indiv[i].position; 
      l++;
    }
  }
  free(relabel);
  iftCompNumIndivsPerGroup(reducedpheno);
  return(reducedpheno);
}

int main(int argc, char *argv[]) 
{
  timer          *t1=NULL,*t2=NULL;
  iftPheno       *pheno=NULL,*reducedpheno=NULL;


  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=3)
    iftError("Usage: iftEliminateSmallGroups <phenotype file> <minimum number of individuals per group>","main");
  
  t1     = iftTic();
  
  pheno        = iftReadPhenotypeInfo(argv[1]);
  
  reducedpheno = iftEliminateSmallGroups(pheno,atoi(argv[2]));
  iftWritePhenotypeInfo(reducedpheno,argv[1]); 

  t2     = iftToc();
  fprintf(stdout,"Small groups eliminated in %f ms\n",iftCompTime(t1,t2));
  
  iftDestroyPheno(&pheno);
  iftDestroyPheno(&reducedpheno); 

  /* ---------------------------------------------------------- */
  

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



