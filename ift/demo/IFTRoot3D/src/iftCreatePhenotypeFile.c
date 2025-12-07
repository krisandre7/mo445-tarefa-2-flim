#include "iftRoot3D.h"

/* 
   Author: Alexandre Falcao.  

   Description: This program takes an option of subpopulation (or all
   subpopulations), reads two files, one with line and subpopulation
   identifications and the other with line and trait values, and
   creates a phenotype file with line identification, subpopulation
   identification, and trait values for the selected subpopulation (or
   for all subpopulations as default). Before the merging, it sorts
   both files by the increasing order of their line identification
   number for merging and output generation. An individual without
   subpopulation is assumed as NA in the phenotype file. It also
   outputs a trait selection file (SelectedTraits.csv) with all traits
   automatically selected. If you want to deselect a trait, you must
   edit this file.  */

/*--------------- Methods that are specific for this module ------------------*/

iftPheno *iftReadIndivSubpop(char *filename); /* Read file with line and subpopulation identifications */ 
iftPheno *iftReadIndivTraits(char *filename); /* Read file with line identification and trait values */
iftPheno *iftMergeIndivInfo(iftPheno *sp, iftPheno *tr, char *subpop); /* Merge line, subpopulation, and traits for a selected subpopulation */     

/*--------------- Their implementation ---------------------------------------*/

iftPheno *iftReadIndivSubpop(char *filename) 
{
  iftPheno *sp=NULL;
  FILE     *fp=NULL;
  int       i,nindivs=0;
  char      buffer[ReadBufferSize];
  char     *token; 
  
  /* Count the number of individuals */

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadIndivSubpop");
  
  nindivs=0;
  while (!feof(fp))
    {
      fgets(buffer,ReadBufferSize,fp);
      nindivs++;
    }
  nindivs -= 2;
  fclose(fp);

  fprintf(stdout,"Total number of individuals in the subpops file: %d\n",nindivs);

  /* Read the file */

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadIndivSubpop");

  sp = iftCreatePheno(nindivs,0);   /* allocate memory */

  fgets(buffer,ReadBufferSize,fp);  /* skip the header */

  for (i=0; i < nindivs; i++) { /* read the information per individual */
    fgets(buffer,ReadBufferSize,fp);     
    token   = strtok(buffer,";"); 
    strncpy(sp->indiv[i].lineID,token,LineIDSize-1);      
    sp->indiv[i].lineID[LineIDSize-1]='\0'; 
    token  = strtok(NULL,"\n");
    iftRemoveCarriageReturn(token);
    strncpy(sp->indiv[i].subpop,token,SubpopNameSize-1);
    sp->indiv[i].subpop[SubpopNameSize-1]='\0';    
  }

  fclose(fp);
  
  return(sp);
}

iftPheno *iftReadIndivTraits(char *filename) 
{
  iftPheno *tr=NULL;
  FILE     *fp=NULL;
  int       i,j,nindivs=0,ntraits=0;
  char      buffer[ReadBufferSize], *token;
  
  /* Count the number of traits */

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadIndivTraits");

  fgets(buffer,ReadBufferSize,fp);
  i=0;
  while ((i<ReadBufferSize)&&(buffer[i]!='\n')){
    if (buffer[i]==';')
      ntraits++;
    i++;
  }

  /* Count the number of individuals */

  nindivs=0;
  while (!feof(fp))
    {
      fgets(buffer,ReadBufferSize,fp);
      nindivs++;
    }
  nindivs -= 1;
  fclose(fp);

  fprintf(stdout,"Total number of individuals in the trait file: %d\n",nindivs);

  /* Read the file */

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadIndivTraits");

  tr = iftCreatePheno(nindivs,ntraits); /* allocate memory */

  /* Read trait names */

  fgets(buffer,ReadBufferSize,fp); 
  token  = strtok(buffer,";"); /* skip LineID in header */  
  for (j=0; j < ntraits-1; j++) {
    token  = strtok(NULL,";");
    strncpy(tr->traitname[j],token,TraitNameSize-1); 
    tr->traitname[j][TraitNameSize-1] ='\0';
  }

  token  = strtok(NULL,"\n");
  iftRemoveCarriageReturn(token);
  strncpy(tr->traitname[j],token,TraitNameSize-1); 
  tr->traitname[j][TraitNameSize-1] ='\0';

  /* Read lines and traits of each individual */

  for (i=0; i < nindivs; i++) {
    fgets(buffer,ReadBufferSize,fp); 
    token   = strtok(buffer,";");
    strncpy(tr->indiv[i].lineID,token,LineIDSize-1); 
    tr->indiv[i].lineID[LineIDSize-1]='\0'; 

    for (j=0; j < ntraits-1; j++) {
      token  = strtok(NULL,";");
      tr->indiv[i].traitvalue[j] = atof(token);
    }
    token  = strtok(NULL,"\n");
    tr->indiv[i].traitvalue[j] = atof(token);

  }

  fclose(fp);
  
  return(tr);
}


iftPheno *iftMergeIndivInfo(iftPheno *sp, iftPheno *tr, char *subpop)
{
  iftPheno *pheno=NULL;
  int i,j,i1,i2,nindivs;
  
  if (strcmp(subpop,"ALL")==0){
    nindivs = tr->nindivs;
  }else{
    /* count the number of individuals from the desired subpopulation */  
    nindivs=0; i1 = i2 = 0;
    while((i1 < sp->nindivs)&&(i2 < tr->nindivs)){
      if (strcmp(sp->indiv[i1].lineID,tr->indiv[i2].lineID)==0){
	if (strcmp(sp->indiv[i1].subpop,subpop)==0)
	  nindivs++;
	i2++;
      }else{
	if (strcmp(sp->indiv[i1].lineID,tr->indiv[i2].lineID) < 0)
	  i1++;
	else
	  i2++;
      }      
    }
    fprintf(stdout,"Number of individuals in subpop %s: %d\n",subpop,nindivs);
  }


  /* Copy the names of the traits */

  pheno = iftCreatePheno(nindivs,tr->ntraits);
  for (j=0; j < pheno->ntraits; j++)
    strcpy(pheno->traitname[j],tr->traitname[j]);

  /* Merge the files */

  if (strcmp(subpop,"ALL")==0){
    i1=0; i2=0; i=0;
    while((i1 < sp->nindivs)&&(i2 < tr->nindivs)&&(i < nindivs)){
      if (strcmp(sp->indiv[i1].lineID,tr->indiv[i2].lineID)==0){
	strcpy(pheno->indiv[i].lineID,tr->indiv[i2].lineID);
	strcpy(pheno->indiv[i].subpop,sp->indiv[i1].subpop);
	for (j=0; j < pheno->ntraits; j++)	
	  pheno->indiv[i].traitvalue[j] = tr->indiv[i2].traitvalue[j];
	pheno->indiv[i].label = tr->indiv[i2].label;
	i2++; i++;
      }else{
	if (strcmp(sp->indiv[i1].lineID,tr->indiv[i2].lineID) < 0) /* line with no measures */
	  i1++;
	else{ /* subpop = NA */
	  strcpy(pheno->indiv[i].lineID,tr->indiv[i2].lineID);
	  sprintf(pheno->indiv[i].subpop,"NA");
	  for (j=0; j < pheno->ntraits; j++)	
	    pheno->indiv[i].traitvalue[j] = tr->indiv[i2].traitvalue[j];
	  pheno->indiv[i].label = tr->indiv[i2].label;
	  i2++; i++;
	}	  
      }
    }
    while ((i2 < tr->nindivs)&&(i < nindivs)){ /* subpop = NA */
      strcpy(pheno->indiv[i].lineID,tr->indiv[i2].lineID);
      sprintf(pheno->indiv[i].subpop,"NA");
      for (j=0; j < pheno->ntraits; j++)	
	pheno->indiv[i].traitvalue[j] = tr->indiv[i2].traitvalue[j];
      pheno->indiv[i].label = tr->indiv[i2].label;
      i2++; i++;
    }   
  }else{ /* For a desired subpopulation */
    i1=0; i2=0; i=0;
    while((i1 < sp->nindivs)&&(i2 < tr->nindivs)&&(i < nindivs)){
      if (strcmp(sp->indiv[i1].lineID,tr->indiv[i2].lineID)==0){
	if (strcmp(sp->indiv[i1].subpop,subpop)==0){
	  strcpy(pheno->indiv[i].lineID,tr->indiv[i2].lineID);
	  strcpy(pheno->indiv[i].subpop,sp->indiv[i1].subpop);
	  for (j=0; j < pheno->ntraits; j++)	
	    pheno->indiv[i].traitvalue[j] = tr->indiv[i2].traitvalue[j];
	  pheno->indiv[i].label = tr->indiv[i2].label;
	  i++;
	}
	i2++; 
      }else{
	if (strcmp(sp->indiv[i1].lineID,tr->indiv[i2].lineID) < 0) /* line with no measures */
	  i1++;
	else{ /* subpop = NA */
	  i2++;
	}	  
      }
    }
  }

  return(pheno);
} 


int main(int argc, char *argv[]) 
{
  timer    *t1=NULL,*t2=NULL;
  iftPheno *tr=NULL,*sp=NULL,*pheno=NULL,*aux=NULL;
  float    *alpha=NULL;
  int       j;
  char      filename[100];

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=2)
    iftError("Usage: iftCreatePhenotypeFile <subpop/ALL>","main");

 
  t1 = iftTic();

  aux = iftReadIndivSubpop("Subpops.csv"); 
  sp  = iftSortIndivByLineID(aux);
  iftDestroyPheno(&aux);
  aux = iftReadIndivTraits("Traits.csv"); 
  tr = iftSortIndivByLineID(aux);
  iftDestroyPheno(&aux);
  pheno = iftMergeIndivInfo(sp,tr,argv[1]);
  sprintf(filename,"Phenotype_%s.csv",argv[1]);
  iftWritePhenotypeInfo(pheno,filename);
  /* select all traits */
  alpha = iftAllocFloatArray(pheno->ntraits);
  for (j=0; j < pheno->ntraits; j++) 
    alpha[j] = 1.0;  
  iftWriteSelectedTraits(pheno,alpha,"SelectedTraits.csv");

  t2 = iftToc();
  fprintf(stdout,"Phenotype file was created in %f ms\n",iftCompTime(t1,t2));

  free(alpha);  
  iftDestroyPheno(&sp);
  iftDestroyPheno(&tr);
  iftDestroyPheno(&pheno);
  

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



