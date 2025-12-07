#include "iftRoot3D.h"

iftPheno *iftCreatePheno(int nindivs, int ntraits)
{
  iftPheno *pheno=(iftPheno *) calloc(1,sizeof(iftPheno));
  int i,j;

  if ((nindivs<=0)||(ntraits < 0))
    iftError("Invalid number of individuals/traits","iftCreatePheno");

  pheno->indiv     = (iftIndiv *)calloc(nindivs,sizeof(iftIndiv));
  
  if (pheno->indiv != NULL) {
    pheno->nindivs = nindivs;
    pheno->ntraits = ntraits;
    pheno->ngroups = 0;
    pheno->nindivs_group = NULL;
    if (ntraits > 0){
      pheno->traitname = (char **) calloc(ntraits,sizeof(char *));
      if (pheno->traitname != NULL) {
	for (j=0; j < ntraits; j++) 
	  pheno->traitname[j] = iftAllocCharArray(TraitNameSize);
      }
      for (i=0; i < nindivs; i++) {
	pheno->indiv[i].traitvalue = iftAllocFloatArray(ntraits);
	pheno->indiv[i].label      = 0;
	pheno->indiv[i].position   = NIL;
      }
    }
  }else{
    iftError(MSG1,"iftCreatePheno");
  }

  return(pheno);
}

void iftDestroyPheno(iftPheno **pheno)
{
  iftPheno *aux=*pheno;
  int i,j;

  if (aux != NULL) {
    if (aux->nindivs_group != NULL) 
      free(aux->nindivs_group);
    for (j=0; j < aux->ntraits; j++) 
      free(aux->traitname[j]);
    free(aux->traitname);
    for (i=0; i < aux->nindivs; i++) 
      free(aux->indiv[i].traitvalue);
    free(aux->indiv);
    free(aux);
    *pheno = NULL;
  }
}

iftPheno *iftCopyPhenotypeInfo(iftPheno *pheno)
{
  iftPheno *copiedpheno=iftCreatePheno(pheno->nindivs,pheno->ntraits);
  int i,j; 
  
  for (j=0; j < pheno->ntraits; j++) 
    strcpy(copiedpheno->traitname[j],pheno->traitname[j]);

  copiedpheno->ngroups = pheno->ngroups;  
  if (pheno->nindivs_group != NULL){
    copiedpheno->nindivs_group = iftAllocIntArray(pheno->ngroups);
    for (i=0; i < copiedpheno->ngroups; i++)
      copiedpheno->nindivs_group[i] = pheno->nindivs_group[i];
  }

  for(i=0; i < pheno->nindivs; i++) {
    strcpy(copiedpheno->indiv[i].lineID,pheno->indiv[i].lineID);
    strcpy(copiedpheno->indiv[i].subpop,pheno->indiv[i].subpop);
    for (j=0; j < pheno->ntraits; j++) 
      copiedpheno->indiv[i].traitvalue[j] = pheno->indiv[i].traitvalue[j]; 
    copiedpheno->indiv[i].label    = pheno->indiv[i].label;     
    copiedpheno->indiv[i].position = pheno->indiv[i].position; 
  }

  
  return(copiedpheno);
}


iftPheno *iftSortIndivByLineID(iftPheno *pheno)
{
  char **lineID;
  int *index=iftAllocIntArray(pheno->nindivs);
  iftPheno *sortedpheno=iftCreatePheno(pheno->nindivs,pheno->ntraits);
  int  i,j;

  /* Allocate memory for the list of lines */

  lineID = (char **) calloc(pheno->nindivs,sizeof(char *));
  for (i=0; i < pheno->nindivs; i++) 
    lineID[i] = iftAllocCharArray(LineIDSize);

  /* Copy the names of the traits */
  
  for (j=0; j < pheno->ntraits; j++) 
    strcpy(sortedpheno->traitname[j],pheno->traitname[j]);

  /* Copy number of groups */

  sortedpheno->ngroups = pheno->ngroups;
  if (pheno->nindivs_group != NULL){
    sortedpheno->nindivs_group = iftAllocIntArray(pheno->ngroups);
    for (i=0; i < sortedpheno->ngroups; i++)
      sortedpheno->nindivs_group[i] = pheno->nindivs_group[i];
  }

  /* Sort by line identifications */

  for (i=0; i < pheno->nindivs; i++){
    strcpy(lineID[i],pheno->indiv[i].lineID);
    index[i] =i;
  } 
  iftSQuickSort(lineID, index, 0, pheno->nindivs-1, INCREASING, LineIDSize);

  /* Copy sorted individuals */

  for (i=0; i < pheno->nindivs; i++){
    strcpy(sortedpheno->indiv[i].lineID,pheno->indiv[index[i]].lineID);
    strcpy(sortedpheno->indiv[i].subpop,pheno->indiv[index[i]].subpop);
    for (j=0; j < pheno->ntraits; j++) {
      sortedpheno->indiv[i].traitvalue[j] = 
	pheno->indiv[index[i]].traitvalue[j];
    } 
    sortedpheno->indiv[i].label    = pheno->indiv[index[i]].label;    
    sortedpheno->indiv[i].position = pheno->indiv[index[i]].position; 
  }

  /* Free memory */

  for (i=0; i < pheno->nindivs; i++) 
    free(lineID[i]);
  free(lineID);
  free(index);

  return(sortedpheno);

}

iftPheno *iftSortIndivByLabel(iftPheno *pheno)
{
  int *label=iftAllocIntArray(pheno->nindivs);
  int *index=iftAllocIntArray(pheno->nindivs);
  iftPheno *sortedpheno=iftCreatePheno(pheno->nindivs,pheno->ntraits);
  int  i,j;

  /* Copy the names of the traits */
  
  for (j=0; j < pheno->ntraits; j++) 
    strcpy(sortedpheno->traitname[j],pheno->traitname[j]);

  /* Copy number of groups */

  sortedpheno->ngroups = pheno->ngroups;
  if (pheno->nindivs_group != NULL){
    sortedpheno->nindivs_group = iftAllocIntArray(pheno->ngroups);
    for (i=0; i < sortedpheno->ngroups; i++)
      sortedpheno->nindivs_group[i] = pheno->nindivs_group[i];
  }

  /* Sort by group label */

  for (i=0; i < pheno->nindivs; i++){
    label[i]=pheno->indiv[i].label;
    index[i] =i;
  } 
  iftBucketSort(label,index,pheno->nindivs,INCREASING);

  /* Copy sorted individuals */

  for (i=0; i < pheno->nindivs; i++){
    strcpy(sortedpheno->indiv[i].lineID,pheno->indiv[index[i]].lineID);
    strcpy(sortedpheno->indiv[i].subpop,pheno->indiv[index[i]].subpop);
    for (j=0; j < pheno->ntraits; j++) {
      sortedpheno->indiv[i].traitvalue[j] = 
	pheno->indiv[index[i]].traitvalue[j];
    } 
    sortedpheno->indiv[i].label    = pheno->indiv[index[i]].label; 
    sortedpheno->indiv[i].position = pheno->indiv[index[i]].position; 
  }

  free(label);
  free(index);

  return(sortedpheno);

}

iftPheno *iftReadPhenotypeInfo(char *filename)  
{
  iftPheno *pheno=NULL;
  int       ntraits,nindivs,i,j;
  char      buffer[ReadBufferSize],*token;
  FILE     *fp=NULL;

  /* Count the number of traits */

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadPhenotypeInfo");

  fgets(buffer,ReadBufferSize,fp);
  i=0; ntraits=0;
  while ((i<ReadBufferSize)&&(buffer[i]!='\n')){
    if (buffer[i]==';')
      ntraits++;
    i++;
  }
  ntraits -= 2;

  /* Count the number of individuals */

  nindivs=0;
  while (!feof(fp))
    {
      fgets(buffer,ReadBufferSize,fp);
      nindivs++;
    }
  nindivs -= 1;

  fclose(fp);

  /* read phenotype information */

  pheno = iftCreatePheno(nindivs,ntraits);

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadPhenotypeInfo");

  fgets(buffer,ReadBufferSize,fp);
  token  = strtok(buffer,";"); /* skip LineID   in header */
  token  = strtok(NULL,";");   /* skip SubpopID in header */

  /* read the names of the traits */

  for (j=0; j < pheno->ntraits; j++){ 
    token  = strtok(NULL,";");
    strncpy(pheno->traitname[j],token,TraitNameSize-1); 
    pheno->traitname[j][TraitNameSize-1] ='\0';
  }


  /* read line id, subpop id, trait values, and label for each individual */

  for(i=0; i < pheno->nindivs; i++) {
    fgets(buffer,ReadBufferSize,fp);
    token  = strtok(buffer,";");
    strncpy(pheno->indiv[i].lineID,token,LineIDSize-1); 
    pheno->indiv[i].lineID[LineIDSize-1] ='\0';
    token  = strtok(NULL,";");
    strncpy(pheno->indiv[i].subpop,token,SubpopNameSize-1); 
    pheno->indiv[i].subpop[SubpopNameSize-1] ='\0';
    for (j=0; j < pheno->ntraits; j++) {
      token  = strtok(NULL,";");
      pheno->indiv[i].traitvalue[j] = atof(token);
    }
    token  = strtok(NULL,"\n");
    pheno->indiv[i].label = atoi(token);
  }

  fclose(fp);

  return(pheno);
}

void iftCompNumIndivsPerGroup(iftPheno *pheno)
{
  int i;

  /* Compute the number of groups */

  pheno->ngroups = 0;
  for(i=0; i < pheno->nindivs; i++) {
    if (pheno->indiv[i].label > pheno->ngroups)
      pheno->ngroups = pheno->indiv[i].label;    
  }

  /* Compute the number of individuals per group */

  if (pheno->nindivs_group==NULL){
    pheno->nindivs_group = iftAllocIntArray(pheno->ngroups);
    for(i=0; i < pheno->nindivs; i++) {
      pheno->nindivs_group[pheno->indiv[i].label-1]++;
    }
  }
  
  printf("\n");
  for(i=0; i < pheno->ngroups; i++) {
    printf("group %d: %d individuals\n",i+1,pheno->nindivs_group[i]); 
  }

}

void iftWritePhenotypeInfo(iftPheno *pheno, char *filename)
{
  FILE *fp=fopen(filename,"w");
  int i,j;
  
  fprintf(fp,"LineID;SubpopID;");

  /* write the names of the traits */

  for (j=0; j < pheno->ntraits; j++) 
    fprintf(fp,"%s;",pheno->traitname[j]);
  fprintf(fp,"Group");
  fprintf(fp,"\n");

  /* write line id, subpop id, trait values, and label of each individual */

  for(i=0; i < pheno->nindivs; i++) {
    fprintf(fp,"%s;",pheno->indiv[i].lineID);
    fprintf(fp,"%s;",pheno->indiv[i].subpop);
    for (j=0; j < pheno->ntraits; j++) 
      fprintf(fp,"%f;",pheno->indiv[i].traitvalue[j]);
    fprintf(fp,"%d",pheno->indiv[i].label);    
    fprintf(fp,"\n");
  }
    
  fclose(fp);

}

void iftPrintPhenotypeInfo(iftPheno *pheno)
{
  int i,j;
  
  fprintf(stdout,"LineID;SubpopID;");

  /* write the names of the traits */

  for (j=0; j < pheno->ntraits; j++) 
    fprintf(stdout,"%s;",pheno->traitname[j]);
  fprintf(stdout,"Group");
  fprintf(stdout,"\n");

  /* write line id, subpop id, trait values, and label of each individual */

  for(i=0; i < pheno->nindivs; i++) {
    fprintf(stdout,"%s;",pheno->indiv[i].lineID);
    fprintf(stdout,"%s;",pheno->indiv[i].subpop);
    for (j=0; j < pheno->ntraits; j++) 
      fprintf(stdout,"%f;",pheno->indiv[i].traitvalue[j]);
    fprintf(stdout,"%d",pheno->indiv[i].label);    
    fprintf(stdout,"\n");
  }
    
}

void iftReadSelectedTraits(char *filename, iftDataSet *Z)
{
  FILE *fp=NULL;
  char  buffer[ReadBufferSize],*token;
  int   j,ntraits;
  
  /* Count the number of traits */

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadSelectedTraits");


  fgets(buffer,ReadBufferSize,fp);
  j=0; ntraits=0;
  while ((j<ReadBufferSize)&&(buffer[j]!='\n')){
    if (buffer[j]==';')
      ntraits++;
    j++;
  }
  ntraits += 1;

  if (ntraits != Z->nfeats)
    iftError("Incompatible number of traits","iftReadSelectedTraits");

  /* read the weight of each trait */

  fgets(buffer,ReadBufferSize,fp);
  token  = strtok(buffer,";");
  for (j=0; j < Z->nfeats-1; j++) {
    Z->alpha[j] = atof(token);
    token  = strtok(NULL,";");
  }

  fclose(fp);

}

void iftWriteSelectedTraits(iftPheno *pheno, float *alpha, char *filename)
{
  FILE *fp=fopen(filename,"w");
  int  j;
  
  /* write the names of the traits */

  for (j=0; j < pheno->ntraits-1; j++) 
    fprintf(fp,"%s;",pheno->traitname[j]);
  fprintf(fp,"%s",pheno->traitname[j]);
  fprintf(fp,"\n");

  /* write the selection factor of each trait */

  for (j=0; j < pheno->ntraits-1; j++) 
    fprintf(fp,"%f;",alpha[j]);
  fprintf(fp,"%f",alpha[j]);

  fclose(fp);

}


iftDataSet *iftPhenoToDataset(iftPheno *pheno){
  iftDataSet *Z=NULL;
  int i, j;

  Z = iftCreateDataSet(pheno->nindivs, pheno->ntraits);
  
  for (j=0; j < Z->nfeats; j++) {      
    Z->alpha[j] = 1.0;
  }

  for (i=0; i < Z->nsamples; i++) {
    for (j=0; j < Z->nfeats; j++) {      
      Z->sample[i].feat[j] = pheno->indiv[i].traitvalue[j];
    }
  }

  return(Z);
}

void iftCopyIndivLabels(iftDataSet *Z, iftPheno *pheno)
{
  int i; 

  pheno->ngroups = Z->nlabels;
  fprintf(stdout,"Number of groups: %d\n",Z->nlabels);
  if (pheno->nindivs_group == NULL){
    pheno->nindivs_group = iftAllocIntArray(pheno->ngroups);
  }else{
    for (i=0; i < pheno->ngroups; i++) {
      pheno->nindivs_group[i]=0;
    }
  }

  for (i=0; i < Z->nsamples; i++) {
    pheno->indiv[i].label = Z->sample[i].label;
    pheno->nindivs_group[pheno->indiv[i].label-1]++;
  }
    
}

iftPheno *iftSelectSubpop(iftPheno *pheno, char *subpop)
{
  iftPheno *newpheno=NULL;
  int i, j, k, nindivs=0;

  if (strcmp(subpop,"ALL")==0)
    return(iftCopyPhenotypeInfo(pheno));

  for (i=0; i < pheno->nindivs; i++) 
    if (strcmp(pheno->indiv[i].subpop,subpop)==0)
      nindivs++;

  if (nindivs==0)
    iftError("Could not find the desired subpopulation","iftSelectSubpop");
 
  newpheno = iftCreatePheno(nindivs,pheno->ntraits);
  for (j=0; j < pheno->ntraits; j++) 
    strcpy(newpheno->traitname[j],pheno->traitname[j]);

  newpheno->ngroups = pheno->ngroups;

  k = 0; 
  for (i=0; i < pheno->nindivs; i++){
    if (strcmp(pheno->indiv[i].subpop,subpop)==0){
      strcpy(newpheno->indiv[k].lineID,pheno->indiv[i].lineID);
      strcpy(newpheno->indiv[k].subpop,pheno->indiv[i].subpop);
      for (j=0; j < pheno->ntraits; j++) {
	newpheno->indiv[k].traitvalue[j] = pheno->indiv[i].traitvalue[j];
      } 
      newpheno->indiv[k].label    = pheno->indiv[i].label; 	
      newpheno->indiv[k].position = pheno->indiv[i].position; 	
      k++;
    }
  }

  return(newpheno);
}

iftGeno *iftCreateGeno(int nsnps, int nindivs, int ngroups)
{
  iftGeno *geno=(iftGeno *)calloc(1,sizeof(iftGeno));
  int i;

  geno->nsnps     = nsnps;
  geno->nindivs   = nindivs;
  geno->ngroups   = ngroups;

  geno->lineID   = (char **) calloc(nindivs,sizeof(char *));
  for (i=0; i < nindivs; i++) 
    geno->lineID[i] = iftAllocCharArray(LineIDSize);

  geno->snp      = (iftSNPs *) calloc(nsnps, sizeof(iftSNPs));
  if (geno->snp == NULL) 
    iftError(MSG2,"iftCreateGeno");
  for (i=0; i < nsnps; i++) {
    geno->snp[i].allele            = iftAllocCharArray(nindivs);
    geno->snp[i].pvalue            = iftAllocLongDoubleArray(ngroups);
    geno->snp[i].min_pvalue_perm   = iftAllocLongDoubleArray(ngroups);
    geno->snp[i].max_pvalue_perm   = iftAllocLongDoubleArray(ngroups);
    geno->snp[i].first_pvalue_perm = iftAllocLongDoubleArray(ngroups);
    geno->snp[i].group_relevance   = iftAllocCharArray(ngroups);
  }
  return(geno);
}

void iftDestroyGeno(iftGeno **geno)
{
  iftGeno *aux=*geno;
  int i;

  if (aux != NULL){    
    for (i=0; i < aux->nindivs; i++) 
      free(aux->lineID[i]);
    free(aux->lineID);
    for (i=0; i < aux->nsnps; i++) {
      free(aux->snp[i].allele);
      free(aux->snp[i].pvalue);
      free(aux->snp[i].min_pvalue_perm);
      free(aux->snp[i].max_pvalue_perm);
      free(aux->snp[i].first_pvalue_perm);
      free(aux->snp[i].group_relevance);
    }
    free(aux->snp);
    free(aux);
    *geno = NULL;
  }
}


iftGeno *iftReadGenotypeInfo(char *filename, int ngroups)
{
  iftGeno *geno=NULL;  
  FILE    *fp;
  char     buffer[ReadBufferSize];
  char    *token;
  char     lineID[LineIDSize];
  int      nindivs,nsnps,i,j,k;
  

  // Count the number of individuals in the first row and the
  // number of snps after the first one

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadGenotypeInfo");

  // counting columns

  fgets(buffer,ReadBufferSize,fp);
  i=0; nindivs=0;
  while ((i<ReadBufferSize)&&(buffer[i]!='\n')){
    if (buffer[i]==';')
      nindivs++;
    i++;
  }
  nindivs -= 2;

  printf("Number of lines %d in the genotype file\n",nindivs);


  // counting rows
  nsnps=0;
  while (!feof(fp)){
    fgets(buffer,ReadBufferSize,fp);
    nsnps++;
  }
  nsnps-=1;

  printf("Number of snps %d in the genotype file\n",nsnps);

  fclose(fp);

  // Read individuals in the first row and snp information in the remaining ones

  geno = iftCreateGeno(nsnps,nindivs,ngroups);

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadGenotypeInfo");

  fgets(buffer,ReadBufferSize,fp);

  /* Jump the first three fields and read lineIDs */ 
  i=k=0;
  for (j=0; j < (strlen(buffer))&&((i-3)<nindivs); j++) {
    if ((buffer[j]!=';')&&(buffer[j]!='\n')&&(buffer[j]!='\r')){
      lineID[k]=buffer[j]; k++;
    }else{
      lineID[k]='\0';
      if (i > 2){
	strcpy(geno->lineID[i-3],lineID); 
      }
      i++;
      k=0;
    }
  }

  /* 
  token=strtok(buffer,";"); // skip SNP ID in header
  token=strtok(NULL,";"); // skip Chromosome in header
  token=strtok(NULL,";"); // skip Position in header
  for (i=0; i < nindivs-1; i++){ 
    token=strtok(NULL,";");
    strncpy(geno->lineID[i],token,LineIDSize-1); 
    geno->lineID[LineIDSize-1]='\0';   
  }
  token=strtok(NULL,"\n");
  strncpy(geno->lineID[i],token,LineIDSize-1); 
  geno->lineID[LineIDSize-1]='\0';   
  */

  for (i=0; i < nsnps; i++) {

    fgets(buffer,ReadBufferSize,fp);

    token=strtok(buffer,";");
    strncpy(geno->snp[i].ID,token,SNPIDSize-1); 
    geno->snp[i].ID[SNPIDSize-1]='\0';         

    token=strtok(NULL,";");
    geno->snp[i].chromosome = atoi(token);

    token=strtok(NULL,";");
    geno->snp[i].position = atoi(token);

    for (j=0; j < nindivs-1; j++){ 
      token=strtok(NULL,";");
      geno->snp[i].allele[j]=atoi(token);

    }
    token=strtok(NULL,"\n");
    geno->snp[i].allele[j]=atoi(token);


  }

  fclose(fp);

  return(geno);
}


iftPheno *iftSyncronizeData(iftGeno *geno, iftPheno *pheno)
{
  int l1,l2, nindivs=pheno->nindivs,ngroups=0,j;
  char found=0;
  iftPheno *syncpheno,*sortedpheno;

  for (l1=0; l1 < pheno->nindivs; l1++) {
    found = 0;
    for (l2=0; l2 < geno->nindivs; l2++) 
      if (strcmp(geno->lineID[l2],pheno->indiv[l1].lineID)==0){
	if (pheno->indiv[l1].label > ngroups) // compute new number of groups
	  ngroups = pheno->indiv[l1].label;
	pheno->indiv[l1].position = l2; // syncronize data
	found=1; 
	break;
      }
    if (found==0) {
      nindivs--;
      pheno->indiv[l1].label=NIL; // mark for removal
    }
  }

  if (nindivs < pheno->nindivs){
    printf("Creating new phenotype data with %d individuals\n",nindivs);
    syncpheno          = iftCreatePheno(nindivs,pheno->ntraits);

    for (j=0; j < pheno->ntraits; j++){ 
      strcpy(syncpheno->traitname[j],pheno->traitname[j]);
    }

    syncpheno->ngroups = ngroups;

    if (pheno->nindivs_group == NULL)
      iftError("You must create groups first","iftSyncronizeData");

    syncpheno->nindivs_group = iftAllocIntArray(syncpheno->ngroups);

    l2 = 0;
    for (l1=0; l1 < pheno->nindivs; l1++) {
      if (pheno->indiv[l1].label!=NIL){ 
	strcpy(syncpheno->indiv[l2].subpop,pheno->indiv[l1].subpop);
	for (j=0; j < pheno->ntraits; j++) 
	  syncpheno->indiv[l2].traitvalue[j] = pheno->indiv[l1].traitvalue[j]; 
	strcpy(syncpheno->indiv[l2].lineID,pheno->indiv[l1].lineID);
	syncpheno->indiv[l2].label=pheno->indiv[l1].label;
	syncpheno->indiv[l2].position=pheno->indiv[l1].position;
	syncpheno->nindivs_group[syncpheno->indiv[l2].label-1]++;
	l2++;
      }
    }

  }else{
    syncpheno = pheno;
  }

  /* Sort individuals by label */

  sortedpheno = iftSortIndivByLabel(syncpheno);

  if (syncpheno != pheno)
    iftDestroyPheno(&syncpheno);

  iftDestroyPheno(&pheno);

  return(sortedpheno);
}

