#include "ift.h"

/* Traits in the original root file */

#define TRSL               1 /* cm   */
#define TRSV               2 /* cm3  */
#define MaxWidth           3 /* cm   */
#define MinWidth           4 /* cm   */
#define MaxDepth           5 /* cm   */
#define MaxWidthByMaxDepth 6 
#define Centroid           7 /* cm   */
#define NumMedianRoots     8
#define NumMaxRoots        9
#define Bushiness         10 
#define TRSSA             11 /* cm2  */ 
#define TRSSAByTRSV       12 /* 1/cm */ 
#define TRSLByTRSSA       13 /* 1/cm */
#define VolDistrib        14 
#define ConvexHull        15 /* cm3  */
#define Solidity          16 
#define SRL               17 
#define TipCount          18 


/* Data structure to analyze 3D root measurements */

typedef struct ift_trait {
  char    name[50];
  float   alpha; // importance from 0 to 1
} iftTrait;

typedef struct ift_3Drootmeasures {
  int     nsftv; 
  char    subpop[25];
  int     batch;
  int     cart;
  int     cylinder;
  int     day;
  float  *traitvalue;
  int     label; 
} ift3DRootMeasures;

typedef struct ift_3Droot {
  ift3DRootMeasures *root;
  int nroots; /* number of root images */
  int ntraits; /* number of traits    */
  iftTrait *trait; /* trait names and importance */
  int ngroups; /* number of groups */
} ift3DRootData;

ift3DRootData *iftCreate3DRootData(int nroots, int ntraits)
{
  int i;

  ift3DRootData *R = (ift3DRootData *)calloc(1,sizeof(ift3DRootData));

  R->nroots = nroots; R->ntraits = ntraits;
  R->root   = (ift3DRootMeasures *) calloc(R->nroots,sizeof(ift3DRootMeasures));
  R->trait  = (iftTrait *) calloc(R->ntraits,sizeof(iftTrait));

  for (i=0; i < R->ntraits; i++) {
    R->trait[i].alpha=1.0;
  }

  sprintf(R->trait[0].name,"TRSL(cm)");
  sprintf(R->trait[1].name,"TRSV(cm3)");
  sprintf(R->trait[2].name,"MaxWidth(cm)");
  sprintf(R->trait[2].name,"MinWidth(cm)");
  sprintf(R->trait[2].name,"MaxDepth(cm)");
  sprintf(R->trait[2].name,"MaxWidthByMaxDepth"); 
  sprintf(R->trait[2].name,"Centroid(cm)");
  sprintf(R->trait[2].name,"NumMedianRoots");
  sprintf(R->trait[2].name,"NumMaxRoots");
  sprintf(R->trait[2].name,"Bushiness"); 
  sprintf(R->trait[2].name,"TRSSA(cm2)");
  sprintf(R->trait[2].name,"TRSSAByTRSV(1/cm)");
  sprintf(R->trait[2].name,"TRSLByTRSSA(1/cm)");
  sprintf(R->trait[2].name,"VolDistrib"); 
  sprintf(R->trait[2].name,"ConvexHull(cm3)");  
  sprintf(R->trait[2].name,"Solidity"); 
  sprintf(R->trait[2].name,"SRL");
  sprintf(R->trait[2].name,"TipCount");

  for (i=0; i < R->nroots; i++) {
    R->root[i].traitvalue   = iftAllocFloatArray(R->ntraits);
    R->root[i].label  = 0;
  }
  
  return(R);
}

void iftDestroy3DRootData(ift3DRootData **R)
{
  ift3DRootData *aux=*R;
  int i;

  if (aux != NULL) {
    for (i=0; i < aux->nroots; i++) {
      free(aux->root[i].traitvalue);
    }
      
    free(aux->root);
    free(aux->trait);
    free(aux);
    *R = NULL;
  }
}

ift3DRootData *iftRead3DRootData(char *basename)
{
  ift3DRootData *R=NULL;
  char       filename[200];
  FILE      *fp;
  char       buffer[1024];
  char      *token;
  int        i,j;

  sprintf(filename,"%s.csv",basename);
  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftRead3DRootData");

  
  i=0;
  while (!feof(fp))
    {
      fgets(buffer,1024,fp);
      i++;
    }
  fclose(fp);

  R = iftCreate3DRootData(i-2,18);

  printf("Number of root records is: %d\n",R->nroots);


  fp = fopen(filename,"r");
  fgets(buffer,1024,fp); /* skip header */
  
  for (i=0; i < R->nroots; i++) {
    fgets(buffer,1024,fp); 
    token=strtok(buffer,",");
    R->root[i].nsftv = atoi(token);
    token=strtok(NULL,",");
    strncpy(R->root[i].subpop,token,24);
    R->root[i].subpop[24]='\0';
    token=strtok(NULL,",");
    R->root[i].batch=atoi(token);
    token=strtok(NULL,",");
    R->root[i].cart=atoi(token);
    token=strtok(NULL,",");
    R->root[i].cylinder=atoi(token);
    token=strtok(NULL,",");
    R->root[i].day=atoi(token);
    for (j=0; j < R->ntraits; j++) {
      token=strtok(NULL,",");
      R->root[i].traitvalue[j]=atof(token);
    }    
  }  
  fclose(fp);

  return(R);
}

void iftWrite3DRootDataGroups(ift3DRootData *R, char *basename)
{
  int i,j,s,ntraits=0;
  FILE *fp;
  char filename[200],*newbasename;
  char spop[10],*pos;
  char spopabrev[4][10],spopname[4][24];
  
  sprintf(spopabrev[0],"TEJ");
  sprintf(spopabrev[1],"TRJ");
  sprintf(spopabrev[2],"AUS");
  sprintf(spopabrev[3],"IND");

  sprintf(spopname[0],"Temperate Japonica");
  sprintf(spopname[1],"Tropical Japonica");
  sprintf(spopname[2],"Aus");
  sprintf(spopname[3],"Indica");


  pos = strrchr(basename,'_') + 1;
  sscanf(pos,"%s",spop);

  if (strcmp("ALL",spop)==0) {

    newbasename = strtok(basename,"_");
    for (s=0; s < 4; s++) {    
      sprintf(filename,"%s_%s_Groups.csv",newbasename,spopabrev[s]);
      fp = fopen(filename,"w");

      fprintf(fp,"%d,",R->ngroups);
      ntraits=0;
      for (j=0; j < R->ntraits; j++) {
	if (R->trait[j].alpha!=0.0){
	  ntraits++;
	}
      }
      fprintf(fp,"%d,",ntraits);

      for (j=0; j < R->ntraits; j++) {
	if (R->trait[j].alpha!=0.0){
	  fprintf(fp,"%d,",j+1);
	}
      }
      fprintf(fp,"\n");
      for (i=0; i < R->nroots; i++) {
	if (strcmp(spopname[s],R->root[i].subpop)==0){
	  fprintf(fp,"%d,",R->root[i].nsftv);
	  fprintf(fp,"%d,",R->root[i].label);
	  for (j=0; j < R->ntraits; j++) {
	    if (R->trait[j].alpha!=0.0){
	      fprintf(fp,"%f,",R->root[i].traitvalue[j]);
	    }
	  }
	  fprintf(fp,"\n");
	}
      }
      fclose(fp);
    }  
  }else {

    sprintf(filename,"%s_Groups.csv",basename);
    fp = fopen(filename,"w");

    // write root data with group labels

    fprintf(fp,"%d,",R->ngroups);

    ntraits=0;
    for (j=0; j < R->ntraits; j++) {
      if (R->trait[j].alpha!=0.0){
	ntraits++;
      }
    }
    fprintf(fp,"%d,",ntraits);

    for (j=0; j < R->ntraits; j++) {
      if (R->trait[j].alpha!=0.0){
	fprintf(fp,"%d,",j+1);
      }
    }
    fprintf(fp,"\n");
    for (i=0; i < R->nroots; i++) {
      fprintf(fp,"%d,",R->root[i].nsftv);
      fprintf(fp,"%d,",R->root[i].label);
      for (j=0; j < R->ntraits; j++) {
	if (R->trait[j].alpha!=0.0){
	  fprintf(fp,"%f,",R->root[i].traitvalue[j]);
	}
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
  }

}

void iftPrint3DRootData(ift3DRootData *R)
{
  int i,j;

  printf("NSFTV#;SubPop;Batch;Cart;Cylinder;Day;");
    for (j=0; j < R->ntraits; j++) {
      if (R->trait[j].alpha!=0.0){
	printf("%s;",R->trait[j].name);
      }
    }

  for (i=0; i < R->nroots; i++) {
    printf("%d ",R->root[i].nsftv);
    printf("%s ",R->root[i].subpop);
    printf("%d ",R->root[i].batch);
    printf("%d ",R->root[i].cart);
    printf("%d ",R->root[i].cylinder);
    printf("%d ",R->root[i].day);
    for (j=0; j < R->ntraits; j++) {
      if (R->trait[j].alpha!=0.0)
	printf("%f ",R->root[i].traitvalue[j]);
    }    
    printf("\n");
  }

}

iftDataSet *ift3DRootDataToDataSet(ift3DRootData *R)
{
  iftDataSet *Z=NULL;
  int i, j;

  Z = iftCreateDataSet(R->nroots, R->ntraits);
  
  for (j=0; j < Z->nfeats; j++) {      
    Z->alpha[j] = R->trait[j].alpha;
  }

  for (i=0; i < Z->nsamples; i++) {
    for (j=0; j < Z->nfeats; j++) {      
      Z->sample[i].feat[j] = R->root[i].traitvalue[j];
    }
  }

  return(Z);
}

void iftUpdate3DRootData(iftDataSet *Z, ift3DRootData *R)
{
  int i,j; 

  R->ngroups = 0;
  for (i=0; i < Z->nfeats; i++) {
    R->trait[i].alpha = Z->alpha[i];
  }
  
  for (i=0; i < Z->nsamples; i++) {
    R->root[i].label = Z->sample[i].label-1;
    for (j=0; j < Z->nfeats; j++) 
      R->root[i].traitvalue[j] = Z->sample[i].feat[j]; 
    if (R->root[i].label+1 > R->ngroups)
      R->ngroups = R->root[i].label+1;
  }

}

/* Unsupervised feature selection by MSPS */

/* Data structure for feature selection */

typedef struct ift_unsupfeatselprob {
  iftDataSet *Z;
  int kmax;
} iftUnsupFeatSelProb;


iftUnsupFeatSelProb *iftCreateUnsupFeatSelProb(iftDataSet *Z, int kmax)
{
  iftUnsupFeatSelProb *prob=(iftUnsupFeatSelProb *)calloc(1,sizeof(iftUnsupFeatSelProb));
  prob->Z    = Z;
  prob->kmax = kmax;

  iftSelectUnsupTrainSamples(Z,1.0);

  return(prob);
}

void iftDestroyUnsupFeatSelProb(iftUnsupFeatSelProb **prob)
{
  if (*prob != NULL){ 
    free(*prob);
    *prob = NULL;
  }
}

float iftVPSFitnessForUnsupFeatSelec(void *problem, float *theta)
{
  float value=0.0;
  int i;
  iftKnnGraph *graph;
  iftUnsupFeatSelProb *prob= (iftUnsupFeatSelProb *) problem;
  iftDataSet *Z=prob->Z;
  
  for (i=0; i < Z->nfeats; i++) 
    prob->Z->alpha[i]=theta[i];

  /* Compute fitness value */

  graph = iftCreateKnnGraph(prob->Z,prob->kmax);
  iftUnsupTrain(graph,iftNormalizedCut);
  value = iftNormalizedCut(graph);
  iftDestroyKnnGraph(&graph);

  return(value);
}

void iftUnsupFeatSelecByVPS(iftDataSet *Z, int kmax)
{
  iftUnsupFeatSelProb *prob;
  iftVPS *vps;
  iftKnnGraph *graph;
  int i, nteams = (int)(2*Z->nfeats+1);

  prob = iftCreateUnsupFeatSelProb(Z,kmax);
  vps = iftCreateVPS(Z->nfeats, nteams, (iftVPSFitnessFunc)iftVPSFitnessForUnsupFeatSelec, prob);
  iftVPSMin(vps);
  iftDestroyUnsupFeatSelProb(&prob);

  for (i=0; i < Z->nfeats; i++){
    Z->alpha[i]=vps->best_team.player[i];
    printf("%f ",Z->alpha[i]);
  }
  printf("\n");
  printf("Fitness %f\n",vps->best_team.fitness);

  iftDestroyVPS(&vps);
  iftSelectUnsupTrainSamples(Z,1.0);
  graph = iftCreateKnnGraph(Z,kmax);
  iftUnsupTrain(graph,iftNormalizedCut);
  iftUnsupClassify(graph,Z);
  iftDestroyKnnGraph(&graph);
}

int main(int argc, char *argv[]) 
{
  timer          *t1=NULL,*t2=NULL;
  ift3DRootData  *R=NULL;
  iftDataSet     *Z=NULL;
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
    iftError("Usage must be: iftTraitGroupingByVPS <basename> <maximum scale (0,1]","main");

  // Reading csv file
  
  t1     = iftTic();
  
  R = iftRead3DRootData(argv[1]);
  Z = ift3DRootDataToDataSet(R);
  iftNormalizeFeatures(Z);
  kmax = (int)(atof(argv[2])*Z->nsamples);
  iftUnsupFeatSelecByVPS(Z,kmax);
  iftUpdate3DRootData(Z,R);

  iftDestroyDataSet(&Z);
  iftWrite3DRootDataGroups(R,argv[1]);
  iftDestroy3DRootData(&R);

  t2     = iftToc();
  fprintf(stdout,"clustering in %f ms\n",iftCompTime(t1,t2));

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



