#include "ift.h"

/* This program performs the genotype-phenotype association:

The idea is to output the identification number of the snp,
chromosome, position in the chromossome, minor-allele(0/2)
minor-allele-frequency, p-values-for-minor-allele, and side of the pdf
used to compute those p-values.

If the minor allele is 0, for instance, then 2s will be the most
likely to happen. This will constitute the null hypothesis H0 that the
number of occurrences k of 2s will be higher than the number of
occurrences of 0s at certain expected proportion (probability
alpha). The pdf for the major allele 2 will be higher between n/2 and
n than between 0 and n/2. In any case, we should accumulate the pdf
values greater than or equal to the number k of observations, whenever
k is above the maximum pdf (expected proportion) value, and accumulate
the pdf values less than or equal to k, whenever k is below the
maximum pdf value. The result will be the p-value --- the probability
of H0 be correct (i.e., the pair (snp,group) is insignificant). By
doing so, when p is very low, we can reject H0. This may happen when
there are too many 0s (k is low) or when there are too many 2s (k is
high) much above the expected value. The same reasoning applies for
the case of 2 being the minor allele. We build the pdf of the major
allele 0 in this case.
 
*/


/* Trait code in the original root file */

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


/* Individuals used in the experiment */

typedef struct ift_individuals {
  int  id;     // Individual identification number
  int  label;  // Group label in [0,c] resulting from the trait cluster analysis
  int  pos;    // Position of this individual in the genotype data structure as 
               // computed by iftSyncronizeData  
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

/* Data structure that indicates the snps of a genome and the alleles
   for each (snp,individual). The frequency (percentage of) allele 2 with
   respect to a set with 2s and 0s, and the p-value of each group must
   be computed for each (snp,group).  */

typedef struct ift_snps {
  char         id[12];    // snp identification 
  char        nucleotide; // nucleotide  
  int         chromossome; // Chromossome    
  int         position;  // Position TIGR-v6 in the chromossome
  char        *allele;    // allele in {0,1,2,3} for each individual, being 0 
                          // and 2 the only ones we will consider for analysis. 
  long double *pvalue;    // p-value that must be found for each group
  float        minor_allele_freq;  //frequency of the minor allele for this snp 
  char         minor_allele; // minor allele for this snp, either 0 or 2
  char        *pdf_side;  // 1 indicates that the pdf value was
                          // obtained from the right side of the pdf, 
                          // because the number of observed values for the 
                          // major allele was higher than the expected
                          // value. 0 indicates the other side. We
                          // have a dinstinct value per group.
  char *group_relevance;  // indicates 1 for groups better than random permutations or 0 otherwise 
  long double *min_pvalue_perm; // minimum p-value during permutations for each group
  char *major_allele; // major allele for each group

} iftSNPs;

/* Data structure that indicates the snps of a genome with the alleles
   for each (snp,individual), the number of groups that result from trait cluster
   analysis, p-values and frequency of allele 2 in each (snp,group),
   and the individuals of the genome. */

typedef struct ift_genotype {
  iftSNPs  *snp;     // snps of a genome       
  int       nsnps;   // number of snps
  int       ngroups; // number of groups found by trait cluster analysis
  int      *individualid;  // individual identification number for each individual of the genome
  int       nindividuals;  // number of individuals
} iftGenotype;

/* Data structure for QTL analysis */
  
typedef struct ift_qtl {
  iftIndividuals **individual; // N selections of random individual grouping sets
  int nindividuals; // number of individuals in each set
  int Nsets; // number of random sets
} iftQTL;

/* Allocates memory for individuals */

iftIndividuals *iftCreateIndividuals(int nindividuals)
{
  iftIndividuals *individual;
  
  individual = (iftIndividuals *)calloc(nindividuals,sizeof(iftIndividuals));
  if (individual==NULL)
    iftError(MSG1,"iftCreateIndividuals");
  
  return(individual);
}

/* Destroys individuals in memory */

void iftDestroyIndividuals(iftIndividuals *individual)
{
  if (individual != NULL) 
    free(individual);
}

/* Allocates memory for the phenotype information */

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

/* Destroys phenotype information in memory */

void iftDestroyPhenotype(iftPhenotype *pheno)
{
  if (pheno!=NULL){
    iftDestroyIndividuals(pheno->individual);
    free(pheno->traitid);
    free(pheno);
  }
}

/* Reads the phenotype information */

iftPhenotype *iftReadPhenotype(char *filename)
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
  }

  fclose(fp);

  return(pheno);
}

/* Allocates memory for the genotype information */

iftGenotype *iftCreateGenotype(int nsnps, int nindividuals, int ngroups)
{
  iftGenotype *geno=(iftGenotype *)calloc(1,sizeof(iftGenotype));
  int i;

  geno->nsnps    = nsnps;
  geno->nindividuals   = nindividuals;
  geno->ngroups  = ngroups;

  geno->individualid  = iftAllocIntArray(nindividuals);
  geno->snp     = (iftSNPs *) calloc(nsnps, sizeof(iftSNPs));
  if (geno->snp == NULL) 
    iftError(MSG2,"iftCreateGenotype");
  for (i=0; i < nsnps; i++) {
    geno->snp[i].allele = iftAllocCharArray(nindividuals);
    geno->snp[i].pvalue = iftAllocLongDoubleArray(ngroups);
    geno->snp[i].pdf_side = iftAllocCharArray(ngroups);
    geno->snp[i].min_pvalue_perm = iftAllocLongDoubleArray(ngroups);
    geno->snp[i].major_allele = iftAllocCharArray(ngroups);
    geno->snp[i].group_relevance = iftAllocCharArray(ngroups);
  }
  return(geno);
}

/* Destroys genotype information in memory */

void iftDestroyGenotype(iftGenotype *geno)
{
  int i;

  if (geno != NULL){
    free(geno->individualid);
    for (i=0; i < geno->nsnps; i++) {
      free(geno->snp[i].allele);
      free(geno->snp[i].pvalue);
      free(geno->snp[i].pdf_side);
      free(geno->snp[i].min_pvalue_perm);
      free(geno->snp[i].major_allele);
      free(geno->snp[i].group_relevance);
    }
    free(geno->snp);
    free(geno);
  }
}

/* Read snp information: nucleotide, chromossome, and position in the
   chromossome */

void iftReadSNPInfo(char *filename, iftGenotype *geno)
{

  typedef struct ift_snpinfo {
    char id[12];
    char nucleotide;
    int  chromossome;
    int  position;
  } iftSNPInfo;

  iftSNPInfo *snp;
  FILE        *fp;
  char         buffer[4096];
  char        *token;
  int          nsnps,i,j;
  

  // Count the number of rows (nsnps) after the first one

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadSNPInfo");

  fgets(buffer,4096,fp);
  nsnps=0;
  while (!feof(fp)){
    fgets(buffer,4096,fp);
    nsnps++;
  }
  nsnps-=1;

  fclose(fp);

  // Verify if the number of snps match

  if (nsnps != geno->nsnps) 
    iftWarning("Number of snps differ","iftReadSNPInfo");

  // Discard the first row

  snp = (iftSNPInfo *)calloc(nsnps,sizeof(iftSNPInfo));

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadSNPInfo");

  fgets(buffer,4096,fp);

  // Read snps

  for (i=0; i < nsnps; i++) {
    fgets(buffer,4096,fp);
    token=strtok(buffer,",");
    strncpy(snp[i].id,token,11); 
    snp[i].id[11]='\0'; 
    token=strtok(NULL,",");
    snp[i].nucleotide = token[0]; 
    token=strtok(NULL,",");
    snp[i].chromossome = atoi(token); 
    token=strtok(NULL,",");
    snp[i].position = atoi(token); 
  }

  fclose(fp);

  /* Copy information into genotype data structure */

  for (i=0; i < nsnps; i++) 
    for (j=0; j < geno->nsnps; j++) 
      if (strcmp(snp[i].id,geno->snp[j].id)==0){
	geno->snp[j].nucleotide  = snp[i].nucleotide;
	geno->snp[j].chromossome = snp[i].chromossome;
	geno->snp[j].position    = snp[i].position;
	break;
      }

  free(snp);
}

/* Reads genotype information */

iftGenotype *iftReadGenotype(char *filename, int ngroups)
{
  iftGenotype *geno=NULL;  
  FILE        *fp;
  char         buffer[4096];
  char        *token;
  int          nindividuals,nsnps,i,j;
  

  // Count the number of columns (nindividuals-1) in the first row and the
  // number of rows (nsnps) after the first one

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadGenotype");

  // counting columns
  fgets(buffer,4096,fp);
  token=strtok(buffer,",\n");
  nindividuals=0;
  while (token!=NULL){
    token=strtok(NULL,",\n");
    nindividuals++;
  }
  nindividuals-=2;

  // counting rows
  nsnps=0;
  while (!feof(fp)){
    fgets(buffer,4096,fp);
    nsnps++;
  }
  nsnps-=1;

  fclose(fp);

  // Read individuals in the first row and snp information in the remaining ones

  geno = iftCreateGenotype(nsnps,nindividuals,ngroups);

  if ((fp = fopen(filename,"r"))==0)
    iftError(MSG2,"iftReadGenotype");

  fgets(buffer,4096,fp);
  token=strtok(buffer,",\n");
  token=strtok(NULL,",\n"); // skip first element
  for (i=0; i < nindividuals; i++){ 
    geno->individualid[i] = atoi(token);
    token=strtok(NULL,",");
  }
  
  for (i=0; i < nsnps; i++) {
    fgets(buffer,4096,fp);
    token=strtok(buffer,",");
    strncpy(geno->snp[i].id,token,11); 
    geno->snp[i].id[11]='\0';   
    token=strtok(NULL,",");
    for (j=0; j < nindividuals; j++){ 
      geno->snp[i].allele[j]=atoi(token);
      token=strtok(NULL,",");
    }
  }

  fclose(fp);

  return(geno);
}



/* Computes the number of individuals per group */

int *iftNumberOfIndividualsPerGroup(iftPhenotype *pheno)
{
  int *nindividuals=iftAllocIntArray(pheno->ngroups), l;

  for (l=0; l < pheno->nindividuals; l++) 
    nindividuals[pheno->individual[l].label]++;

  return(nindividuals);
}

/* Creates N selections of random individual groupings */

iftQTL *iftCreateQTL(iftPhenotype *pheno, int Nsets)
{
  float *value=iftAllocFloatArray(pheno->nindividuals);
  int   *index=iftAllocIntArray(pheno->nindividuals), i, l, *n,g,lo,lf;
  iftQTL *rgroups=(iftQTL *)calloc(1,sizeof(iftQTL));
  
  if (pheno->individual[0].pos == NIL) 
    iftError("You must run iftSyncronizeData first","iftCreateQTL");

  rgroups->individual = (iftIndividuals **)calloc(Nsets,sizeof(iftIndividuals));
  if (rgroups == NULL) 
    iftError(MSG2,"iftCreateQTL");
  for (i=0; i < Nsets; i++) 
    rgroups->individual[i] = iftCreateIndividuals(pheno->nindividuals);

  iftRandomSeed(IFT_RANDOM_SEED);
  
  n = iftNumberOfIndividualsPerGroup(pheno);

  for (i=0; i < Nsets; i++) {
    // initialize sorting vectors 
    for (l=0; l < pheno->nindividuals; l++) {
      value[l] = (float)((double) rand () / ((double) RAND_MAX));
      index[l] = l;
    }
    // sort value vector and scramble index vector
    iftFQuickSort(value, index, 0, pheno->nindividuals-1, INCREASING); 
    // copy original individuals from their position in the index vector
    for (l=0; l < pheno->nindividuals; l++) {
      rgroups->individual[i][l].id    = pheno->individual[index[l]].id;
      rgroups->individual[i][l].label = pheno->individual[index[l]].label;
      rgroups->individual[i][l].pos   = pheno->individual[index[l]].pos;
    }
    // relabel the individuals to form the new grouping
    lo=0; lf=0;
    for (g=0; g < pheno->ngroups; g++) {
      lf += n[g];
      for (l=lo; l < lf; l++) 
	rgroups->individual[i][l].label = g;
      lo = lf;    
    }
  }
  rgroups->Nsets = Nsets;
  rgroups->nindividuals = pheno->nindividuals;

  free(value);
  free(index);
  free(n);

  return(rgroups);
}

/* Destroy random groups in memory */

void iftDestroyQTL(iftQTL *rgroups)
{
  int i;

  if (rgroups != NULL) {
    for (i=0; i < rgroups->Nsets; i++) 
      free(rgroups->individual[i]);
    free(rgroups->individual);
    free(rgroups);
  }    
}

/* Verifies if all individuals in the phenotype data are found among
   the individuals in the genotype data and syncronizes these data
   structures. This may imply in deleting individuos from the
   phenotype data which do not have genotype information */

iftPhenotype *iftSyncronizeData(iftGenotype *geno, iftPhenotype *pheno)
{
  int i,l1,l2, nindividuals=pheno->nindividuals;
  char found=0;
  iftPhenotype *spheno;


  for (l1=0; l1 < pheno->nindividuals; l1++) {
    found = 0;
    for (l2=0; l2 < geno->nindividuals; l2++) 
      if (geno->individualid[l2]==pheno->individual[l1].id){
	found=1; 
	break;
      }
    if (!found) {
      nindividuals--;
      pheno->individual[l1].label=NIL; // mark for removal
    }
  }

  if (nindividuals < pheno->nindividuals){
    spheno          = iftCreatePhenotype(nindividuals,pheno->ntraits);
    spheno->ngroups = pheno->ngroups;
    for (i=0; i < pheno->ntraits; i++) {
      spheno->traitid[i] = pheno->traitid[i];
    }

    l2 = 0;
    for (l1=0; l1 < pheno->nindividuals; l1++) {
      if (pheno->individual[l1].label!=NIL){ 
	spheno->individual[l2].id=pheno->individual[l1].id;
	spheno->individual[l2].label=pheno->individual[l2].label;
	l2++;
      }
    }

    iftDestroyPhenotype(pheno);

  }else{
    spheno = pheno;
  }

  /* syncronize data */

  for (l1=0; l1 < spheno->nindividuals; l1++) {
    for (l2=0; l2 < geno->nindividuals; l2++) 
      if (geno->individualid[l2]==spheno->individual[l1].id){
	spheno->individual[l1].pos = l2;
	break;
      }
  }

  return(spheno);
}

/* Find the point of maximum pdf value */

int iftExpectedValue(long double *pdf, int n)
{
  long double Fmax=-INFINITY_LDBL;
  int k,Ek=NIL;
  
  for (k=0; k <= n; k++) 
    if (Fmax < pdf[k]){
      Fmax = pdf[k];
      Ek = k;
    }
  return(Ek);
}

/* Compute the significance level for k >= k0, when k0 >= Ek (the
   expected value), or for k <= k0, when k0 < Ek. */

long double iftSignificanceLevel(long double *pdf, int n, int k0, char *pdf_side)
{
  long double p=0.0; 
  int k, Ek=iftExpectedValue(pdf,n);

  if (k0 >= Ek){
    *pdf_side=1;
    for (k=k0; k <= n; k++)
      p += pdf[k];
  }else{
    *pdf_side=0;
    for (k=0; k <= k0; k++) 
      p += pdf[k];
  }
  
  return(p);
}

/* Normal distribution with mean and variance determined by n and
   alpha. */

long double *iftNormalDistribution(int n, float alpha)
{
  long double  mean = n*alpha, var = n*alpha*(1.0-alpha);
  long double *pdf = iftAllocLongDoubleArray(n+1);
  int k;

  if ((n <= 0)||(alpha < 0.0)||(alpha > 1.0))
    iftError("alpha must be between 0 and 1, and n > 0","iftNormalDistribution");

  if (alpha==0.0){
    pdf[0]=1.0;
    return(pdf);
  }else{
    if (alpha==1.0){
      pdf[n]=1.0;
      return(pdf);
    }
  }

  for (k=0; k <= n; k++) 
    pdf[k] = 1.0/sqrtl(2*PI*var)*expl(-((k-mean)*(k-mean))/(2.0*var));
  
  return(pdf);
}

/* When n*alpha > 10 and n*(1-alpha) > 10, they claim that you can use
   the normal distribution with mean = n*alpha and stdev =
   sqrt(n*alpha*(1-alpha)). This helps because the factorial
   computation generates overflow when n is high. Do not buy that,
   only use normal distribution if the program detects overflow.  */
 
long double *iftBinomialDistribution(int n, float alpha)
{
  long double C, A, B;
  long double *pdf = iftAllocLongDoubleArray(n+1);
  int k;
  long double Fact_n;
 
  if ((n <= 0)||(alpha < 0.0)||(alpha > 1.0)||(n > iftFactorialLimit()))
    iftError("n or alpha are out of the allowed range","iftBinomialDistribution");
  
  if (alpha==0.0){
    pdf[0]=1.0;
    return(pdf);
  }else{
    if (alpha==1.0){
      pdf[n]=1.0;
      return(pdf);
    }
  }

  Fact_n = iftFactorial(n);

  for (k=0; k <= n; k++) {
    C  = ((Fact_n/iftFactorial(k))/iftFactorial(n-k));
    A  = powl(alpha,k);
    B  = powl((1.0-alpha),(n-k));
    pdf[k] = (C*A*B);
  }

  return(pdf);
}


/* Compute the significance levels for all pairs (snp,group). Assuming that the individuals are ordered by groups from label=0,1,...,ngroups. */

void iftFindSignificanceLevels(iftGenotype *geno, iftPhenotype *pheno)
{
  int s,l,g,*nindividuals=iftNumberOfIndividualsPerGroup(pheno),lo,lf;
  int pos, n0, n2, n, limit=iftFactorialLimit(),N0,N2;
  long double *pdf;

  /* computes p-values for each pair (snp,group) */

  for (s=0; s < geno->nsnps; s++) {

    /* Compute the minor allele and the minor allele frequency for the snp */

    N0=N2=0;
    for (l=0; l < pheno->nindividuals; l++) {
      pos = pheno->individual[l].pos;
      if (geno->snp[s].allele[pos]==0){
	N0++;
      }else{ 
	if (geno->snp[s].allele[pos]==2)
	  N2++;
      }
    }
    
    if (N0 < N2){
      geno->snp[s].minor_allele = 0;
      geno->snp[s].minor_allele_freq = (float)N0/(float)(N0+N2);
    }else{
      geno->snp[s].minor_allele = 2;
      geno->snp[s].minor_allele_freq = (float)N2/(float)(N0+N2);
    }

    /* compute p-value for each group in that snp */

    lo = lf = 0;
    for (g=0; g < geno->ngroups; g++) {
      
      lf += nindividuals[g]; 
      
      /* compute the number of alleles in {0,2} for this group */
      
      n0=0; n2=0; 
      for (l=lo; l < lf; l++) {     
	pos  = pheno->individual[l].pos;
	if (geno->snp[s].allele[pos]==0){
	  n0++;
	}else{
	  if (geno->snp[s].allele[pos]==2)
	    n2++;
	}
      }
      n = n0+n2;

      /* generate pdf and compute the p-value of the pair (snp,group)
	 for allele 2 whenever the minor allele is 0, and for
	 allele 0, whenever the minor allele is 2. */

      if (n > 0) {
	if(geno->snp[s].minor_allele_freq > 0.0){
	  if (n <= limit){
	    pdf = 
	      iftBinomialDistribution(n,1.0-geno->snp[s].minor_allele_freq);
	  }else{
	    pdf = 
	      iftNormalDistribution(n,1.0-geno->snp[s].minor_allele_freq);
	  }
	  if (geno->snp[s].minor_allele==0){
	    geno->snp[s].pvalue[g] = iftSignificanceLevel(pdf,n,n2,&geno->snp[s].pdf_side[g]);
	  }else{
	    geno->snp[s].pvalue[g] = iftSignificanceLevel(pdf,n,n0,&geno->snp[s].pdf_side[g]);
	  }
	  free(pdf);
	}else{
	  geno->snp[s].pvalue[g]=1.0; // eliminate pair (snp,group)
	}
      }else{
	geno->snp[s].pvalue[g]=1.0; // eliminate pair (snp,group)
      }	    

      lo = lf;
    }
  }
  free(nindividuals);
}

/* For each (snp,group), it creates sets of random permutations of
   individuals and verifies the position of the real p-value with respect to
   the p-values obtained from those permutations. The pair (snp,group)
   is maintained as significant, whenever the real p-value is among
   the 5% smallest ones. Assuming that individuals are ordered by groups
   from label=0,1,..,ngroups-1. */

void iftQTLAnalysis(iftGenotype *geno, iftPhenotype *pheno, int Nsets, float tolerance, float threshold)
{
  iftQTL      *qtl;
  int          g,l,i,s,pos,*nindividuals=iftNumberOfIndividualsPerGroup(pheno);
  long double *pdf,pvalue;
  int          n0, n2, n, limit=iftFactorialLimit();
  int          lo, lf,L=(int)MAX((tolerance*Nsets),1),nlosses;
  char         pdf_side;
  
  /* generate Nsets random permutations of groups */
  printf("Generating random sets of grouping solutions\n");
  qtl=iftCreateQTL(pheno,Nsets);

  /* compute the QTL analysis */
  printf("Computing the QTL analysis\n");

  for (s=0; s < geno->nsnps; s++) {


    printf("SNP %s (%f perc.)\n",geno->snp[s].id,(float)s/geno->nsnps*100.0);
    lo = lf = 0;
    for (g=0; g < geno->ngroups; g++) {

      // apply threshold 

      if (geno->snp[s].pvalue[g] <= threshold) { 

	geno->snp[s].min_pvalue_perm[g] = 1.0;
	geno->snp[s].group_relevance[g] = 1; 

	lf += nindividuals[g]; nlosses=0;

	for (i=0; (i < qtl->Nsets)&&(nlosses < L); i++) {

	  /* compute the number of alleles in {0,2} for this group */
	
	  n0=0; n2=0; 
	  for (l=lo; l < lf; l++) {     
	    pos      = qtl->individual[i][l].pos;
	    if (geno->snp[s].allele[pos]==0){
	      n0++;
	    }else{
	      if (geno->snp[s].allele[pos]==2)
	      n2++;
	    }
	  }

	  if (n0 > n2) 
	    geno->snp[s].major_allele[g]    = 0;
	  else
	    geno->snp[s].major_allele[g]    = 2;

	  n = n0+n2;
	  
	  /* generate pdf and compute the p-value of the pair (snp,group)
	     for allele 2 whenever the minor allele is 0, and for
	     allele 0, whenever the minor allele is 2. */
	  
	  if (n > 0) {
	    if(geno->snp[s].minor_allele_freq > 0.0){
	      if (n <= limit){
		pdf = 
		  iftBinomialDistribution(n,1.0-geno->snp[s].minor_allele_freq);
	      }else{
		pdf = 
		  iftNormalDistribution(n,1.0-geno->snp[s].minor_allele_freq);
	      }
	      if (geno->snp[s].minor_allele==0){
		pvalue = iftSignificanceLevel(pdf,n,n2,&pdf_side);
	      }else{
		pvalue = iftSignificanceLevel(pdf,n,n0,&pdf_side);
	      }	    

	      if (pvalue < geno->snp[s].min_pvalue_perm[g]) 
		geno->snp[s].min_pvalue_perm[g] = pvalue;

	      if (pvalue < geno->snp[s].pvalue[g])
		nlosses++;

	      free(pdf);
	    }
	  }
	}
     
	if (nlosses >= L)
	  geno->snp[s].group_relevance[g]=0; // this pair (snp,group) is not relevant
           
	lo = lf;
      }
      else
	geno->snp[s].group_relevance[g]=0; // this pair (snp,group) is not relevant 
    }
  }

  iftDestroyQTL(qtl);
  free(nindividuals);
}

void iftWriteSNPSignificLevels(iftGenotype *geno, iftPhenotype *pheno, char *filename)
{
  int   s,*n=iftNumberOfIndividualsPerGroup(pheno);
  FILE *fp;

  fp = fopen(filename,"w");
  
  fprintf(fp,"ID,Chromossome,Position,Minor Allele,Minor Allele Frequency, Lower Tail, Lower Tail, Lower Tail, Lower Tail, Higher Tail, Higher Tail, Higher Tail, Higher Tail"); 
  fprintf(fp,"\n");
  fprintf(fp,"  ,  , , , ,Number of Individuals, Major Allele, p-Value, Minimum p-Value during Permutation, Number of Individuals, Major Allele, p-Value, Minimum p-Value during Permutation");
  fprintf(fp,"\n");

  for (s=0; s < geno->nsnps; s++) {
    fprintf(fp,"%s,",geno->snp[s].id);
    fprintf(fp,"%d,",geno->snp[s].chromossome);
    fprintf(fp,"%d,",geno->snp[s].position);
    fprintf(fp,"%d,",geno->snp[s].minor_allele);
    fprintf(fp,"%f,",geno->snp[s].minor_allele_freq);
    fprintf(fp,"%d,",n[0]);
    fprintf(fp,"%d,",geno->snp[s].major_allele[0]);
    fprintf(fp,"%lle,",geno->snp[s].pvalue[0]);
    if (geno->snp[s].group_relevance[0]==1)
      fprintf(fp,"%lle,",geno->snp[s].min_pvalue_perm[0]);
    else
      fprintf(fp,"%lle,",1.0);
    fprintf(fp,"%d,",n[2]);
    fprintf(fp,"%d,",geno->snp[s].major_allele[2]);
    fprintf(fp,"%lle,",geno->snp[s].pvalue[2]);
    if (geno->snp[s].group_relevance[2]==1)
      fprintf(fp,"%lle,",geno->snp[s].min_pvalue_perm[2]);
    else
      fprintf(fp,"%lle,",1.0);
    fprintf(fp,"\n");
  }
  free(n);
  fclose(fp);
}

/* void iftWriteSNPSignificLevels(iftGenotype *geno, char *filename) */
/* { */
/*   int   g,s,gmin,side0,side1; */
/*   FILE *fp; */

/*   fp = fopen(filename,"w"); */
  

/* number of individuals in lower tail, major allele for lower tail, p-value lower tail, random permutation p-value for lower tail, number of individuals in upper tail, major allele for upper tail, p-value for upper tail, random permutation p-value for upper tail. */

/*   fprintf(fp,"ID,Chromossome,Position,Minor Allele,Minor Allele Frequency, Lower Tail, Lower Tail, Lower Tail, Lower Tail, Higher Tail, Higher Tail, Higher Tail, Higher Tail");  */
/*   fprintf(fp,"\n"); */
/*   fprintf(fp,"  ,  , , , , Number of Individuals, Major Allele, p-Value, Minimum p-Value during Permutation, Major Allele, p-Value, Minimum p-Value during Permutation"); */
/*   fprintf(fp,"\n"); */

/*   for (s=0; s < geno->nsnps; s++) { */

/*     gmin=0; */
/*     for (g=1; g < geno->ngroups; g++) { */
/*       if (geno->snp[s].pvalue[g] < geno->snp[s].pvalue[gmin]){ */
/* 	gmin = g; */
/*       } */
/*     } */

/*     side0 = 0; side1=0; */

/*   fprintf(fp,"ID,Chromossome,Position,Minor Allele,Minor Allele Frequency,Min. P-value,Min. P-Value during permutation"); */
/*   fprintf(fp,"\n"); */

/*   for (s=0; s < geno->nsnps; s++) { */

/*     gmin=0; */
/*     for (g=1; g < geno->ngroups; g++) { */
/*       if (geno->snp[s].pvalue[g] < geno->snp[s].pvalue[gmin]){ */
/* 	gmin = g; */
/*       } */
/*     } */

/*     side0 = 0; side1=0; */
/*     for (g=0; g < geno->ngroups; g++) { */
/*       if ((geno->snp[s].pvalue[g] != 1.0)&&(geno->snp[s].pdf_side[g]==0)) */
/* 	side0+=1; */
/*       if ((geno->snp[s].pvalue[g] != 1.0)&&(geno->snp[s].pdf_side[g]==1)) */
/* 	side1+=1; */
/*     } */
/*     /\*         */
/*     if (!((side0>=1)&&(side1>=1))){ */
/*       geno->snp[s].pvalue[gmin]=1.0; */
/*       geno->snp[s].min_pvalue_perm=1.0; */
/*     } */
/*     *\/   */
/*     // if (((side0>=1)&&(side1>=1))){ */
/*       fprintf(fp,"%s,",geno->snp[s].id); */
/*       fprintf(fp,"%d,",geno->snp[s].chromossome); */
/*       fprintf(fp,"%d,",geno->snp[s].position); */
/*       fprintf(fp,"%d,",geno->snp[s].minor_allele); */
/*       fprintf(fp,"%f,",geno->snp[s].minor_allele_freq); */
/*       fprintf(fp,"%lle,",geno->snp[s].pvalue[gmin]); */
/*       fprintf(fp,"%lle,",geno->snp[s].min_pvalue_perm); */
/*       fprintf(fp,"\n"); */
/*       // } */
/*   } */
/*   fclose(fp); */
/* } */


/* void iftWriteSNPSignificLevels(iftGenotype *geno, char *filename) */
/* { */
/*   int   g,s; */
/*   FILE *fp; */
/*   char  eliminate_snp; */

/*   fp = fopen(filename,"w"); */
  
/*   fprintf(fp,"ID,Chromossome,Position,Minor Allele,Minor Allele Frequency,"); */
/*   for (g=0; g < geno->ngroups; g++)  */
/*     if (g != 1) */
/*       fprintf(fp,"Group %d (p-value), PDF Side,",g); */

/*   fprintf(fp,"\n"); */

/*   for (s=0; s < geno->nsnps; s++) { */

/*     eliminate_snp = 1; */
     
/*     for (g=0; g < geno->ngroups; g++) {       */
/*       if (geno->snp[s].pvalue[g]!=1.0){ */
/* 	eliminate_snp=0; */
/*       } */
/*     } */
    
/*     if (!eliminate_snp){ */
/*       fprintf(fp,"%s,",geno->snp[s].id); */
/*       fprintf(fp,"%d,",geno->snp[s].chromossome); */
/*       fprintf(fp,"%d,",geno->snp[s].position); */
/*       fprintf(fp,"%d,",geno->snp[s].minor_allele); */
/*       fprintf(fp,"%f,",geno->snp[s].minor_allele_freq); */
      
/*       for (g=0; g < geno->ngroups; g++) {       */
/* 	if (g != 1){  */
/* 	  if (geno->snp[s].pvalue[g]!=1.0){ */
/* 	    fprintf(fp,"%lle,",geno->snp[s].pvalue[g]); */
/* 	    fprintf(fp,"%d,",geno->snp[s].pdf_side[g]); */
/* 	  }else{ */
/* 	    fprintf(fp,","); */
/* 	    fprintf(fp,","); */
/* 	  } */
/* 	} */
/*       } */
/*       fprintf(fp,"\n"); */
/*     } */
/*   } */
  
      
/*   fclose(fp); */
/* } */

int main(int argc, char *argv[]) 
{
  timer          *t1=NULL,*t2=NULL;
  iftPhenotype   *pheno=NULL;
  iftGenotype    *geno=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=8)
    iftError("Usage: iftSingleTraitAnalysis <genotype.csv> <snpinfo.csv> <phenotype.csv> <output.csv> <number of permutations> <tolerance> <p-value threshold>","main");

  // Read csv files

  pheno  = iftReadPhenotype(argv[3]); 
  geno   = iftReadGenotype(argv[1],pheno->ngroups);
  iftReadSNPInfo(argv[2], geno);

  t1     = iftTic();
  printf("Syncronizing genotype and phenotype data\n");
  pheno = iftSyncronizeData(geno,pheno);
  printf("Finding significant levels\n");
  iftFindSignificanceLevels(geno,pheno);
  printf("Performing QTL analysis\n");
  iftQTLAnalysis(geno,pheno,atoi(argv[5]),atof(argv[6]),atof(argv[7])); 
  t2     = iftToc();
  fprintf(stdout,"root analysis in %f ms\n",iftCompTime(t1,t2));

  iftWriteSNPSignificLevels(geno, pheno, argv[4]);

  iftDestroyPhenotype(pheno);
  iftDestroyGenotype(geno);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



