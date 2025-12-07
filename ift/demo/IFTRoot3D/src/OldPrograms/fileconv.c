#include "ift.h"

/* Individuals used in the experiment */

typedef struct ift_individuals {
  int  id;     // Individual identification number
  int  label;  // Group label in [0,c] resulting from the trait cluster analysis
  int  pos;    // Position of this individual in the genotype data structure as 
               // computed by iftSyncronizeData  
  float traitvalue; // trait value
} iftIndividuals;

/* Data structure that indicates the individuals, the traits, and the
   resulting number of groups used in a given experiments. */
 
typedef struct ift_phenotype {
  iftIndividuals *individual;  // individuals used in the experiment
  int       nindividuals; // number of individuals
  int       ngroups;     // number c+1 of groups
  int      *traitid;     // trait identification number according to the list
                         // in the beginning of this program
  int       ntraits;     // number of traits
} iftPhenotype;

iftIndividuals *iftCreateIndividuals(int nindividuals)
{
  iftIndividuals *individual;
  
  individual = (iftIndividuals *)calloc(nindividuals,sizeof(iftIndividuals));
  if (individual==NULL)
    iftError(MSG1,"iftCreateIndividuals");
  
  return(individual);
}

iftPhenotype *iftCreatePhenotype(int nindividuals, int ntraits)
{
  iftPhenotype *pheno=(iftPhenotype *)calloc(1,sizeof(iftPhenotype));  
  int i;

  pheno->individual = iftCreateIndividuals(nindividuals);
  pheno->traitid = iftAllocIntArray(ntraits);
  if (pheno->traitid==NULL)
    iftError(MSG1,"iftCreatePhenotype");

  for (i=0; i < nindividuals; i++) {
    pheno->individual[i].pos = NIL;
  }

  pheno->nindividuals  = nindividuals;
  pheno->ntraits = ntraits;

  return(pheno);
}

iftPhenotype *iftReadSingleTrait(char *filename)
{  
  iftPhenotype *pheno=NULL;
  FILE         *fp;
  char          buffer[256];
  char         *token;
  int           l,nindividuals,ntraits=1;

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadPhenotype");
  
  // Count the number of rows (nindividuals)
  nindividuals=0;
  while (!feof(fp))
    {
      fgets(buffer,256,fp);
      nindividuals++;
    }
  nindividuals-=2;
  fclose(fp);

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadPhenotype");

  pheno= iftCreatePhenotype(nindividuals,ntraits);
 
  // This part must change in the file format
  // to support groups from multiple traitures

  fgets(buffer,256,fp);
  token   = strtok(buffer,",");
  pheno->traitid[0] = atoi(token); 
  pheno->ngroups   = 3;

  for (l=0; l < pheno->nindividuals; l++) {
    fgets(buffer,256,fp); 
    token = strtok(buffer,",");
    pheno->individual[l].id=atoi(token);
    token=strtok(NULL,",");
    pheno->individual[l].label=atoi(token);
    token=strtok(NULL,",");
    pheno->individual[l].traitvalue=atof(token);
  }

  fclose(fp);

  return(pheno);
}

void iftWriteSingleTrait(iftPhenotype *pheno, char *filename)
{
  int i,ntraits=1;
  FILE *fp;

  fp = fopen(filename,"w");


  fprintf(fp,"%d;",pheno->ngroups);

  fprintf(fp,"%d;",ntraits);

  fprintf(fp,"%d;",pheno->traitid[0]);
  fprintf(fp,"\n");
  
  for (i=0; i < pheno->nindividuals; i++) {
    fprintf(fp,"%d;",pheno->individual[i].id);
    fprintf(fp,"%d;",pheno->individual[i].label);
    fprintf(fp,"%f;",pheno->individual[i].traitvalue);
    fprintf(fp,"\n");
  }
  
  fclose(fp);
}

int main(int argc, char *argv[]) 
{
  iftPhenotype   *pheno=NULL;

  if (argc!=3)
    iftError("Usage: fileconv <singletrait.csv> <output.csv>","main");

  pheno  = iftReadSingleTrait(argv[1]); 
  iftWriteSingleTrait(pheno,argv[2]);

  return(0);
}
