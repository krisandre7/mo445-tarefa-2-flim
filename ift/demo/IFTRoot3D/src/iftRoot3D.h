#ifndef IFT_ROOT3D_H_
#define IFT_ROOT3D_H_

#include "ift.h"

#define SubpopNameSize   10
#define TraitNameSize    100
#define ReadBufferSize   10000
#define SNPIDSize        20
#define LineIDSize       30


/*------------ Data structures to store phenotype information ---------------*/

typedef struct ift_indiv {
  char   lineID[LineIDSize];     /* line identification */
  char   subpop[SubpopNameSize]; /* subpopulation identification */
  float *traitvalue;             /* trait values */
  int    label;                  /* group label (0 before grouping and from
			            1,2,..., to N, after grouping) */
  int    position;               /* Position of this individual in the
				    genotype data structure, as
				    computed by iftSyncronizeData. */
} iftIndiv;
 
typedef struct ift_pheno {
  iftIndiv *indiv;        /* list of individuals */
  int     nindivs;        /* total number of individuals */
  int     ntraits;        /* number of traits */
  int     ngroups;        /* number of groups */
  int    *nindivs_group;  /* number of individuals per group */
  char  **traitname;      /* name of the traits */
} iftPheno;


/*------------- Data structure to store genotype information ----------------*/ 

typedef struct ift_snps {
  char        ID[SNPIDSize];      /* snp identification */ 
  int         chromosome;         /* chromosome */    
  int         position;           /* position TIGR-v6 in the chromossome */
  char       *allele;             /* allele in {0,1,2,3} for each
                                     individual, being 0 and 2 the
                                     only ones we will consider for
                                     analysis (recessive and dominant
                                     homozygotes). */
  long double *pvalue;            /* p-value that must be found for each group */
  char         minor_allele;      /* minor allele for this snp, either 0 or 2 */
  float        minor_allele_freq; /* frequency of the minor allele */
  char        *group_relevance;   /* indicates 1 for groups better
				     than random permutations or 0
				     otherwise */
  long double *min_pvalue_perm;   /* minimum p-value during
				     permutations for each group */
  long double *max_pvalue_perm;   /* maximum p-value among the best X%
				     (e.g., tolerance X%=5%) during
				     permutations for each group */
  long double *first_pvalue_perm; /* first p-value during permutations
				     for each group */
} iftSNPs;

typedef struct ift_geno {
  iftSNPs  *snp;           /* snps of a genome */       
  int       nsnps;         /* number of snps */
  int       ngroups;       /* number of groups found by trait grouping */
  char    **lineID;        /* line identification number for each
			      individual of the genome */
  int       nindivs;       /* number of individuals */
} iftGeno;


/* -------------------- Methods to manipulate phenotype information -----------*/

iftPheno  *iftCreatePheno(int nindivs, int ntraits); /* Allocate
						       memory to store
						       phenotype
						       information */
void       iftDestroyPheno(iftPheno **pheno);        /* Deallocate
						       memory with
						       phenotype
						       information */ 
iftPheno   *iftSortIndivByLineID(iftPheno *pheno);    /* Sort
						       individuals by
						       line
						       identification */
iftPheno   *iftSortIndivByLabel(iftPheno *pheno);    /* Sort
						       individuals by
						       label */
iftPheno   *iftCopyPhenotypeInfo(iftPheno *pheno); /* Copy phenotype
						    information in */
iftPheno   *iftReadPhenotypeInfo(char *filename);  /* Read line id.,
						    subpopulation id.,
						    trait values, and
						    group label */
void        iftWritePhenotypeInfo(iftPheno *pheno, char *filename);  /* Write line id., subpopulation id., trait values, and group label */

void iftPrintPhenotypeInfo(iftPheno *pheno); /* Print line id., subpopulation id., trait values, and group label */

void        iftReadSelectedTraits(char *filename, iftDataSet *Z);  /* Read file with the trait selection and set the weights [0,1]
  to each trait in Z */ 

void        iftCompNumIndivsPerGroup(iftPheno *pheno); /* Compute the number of individuals per group. It must be called in iftGWAA.c after you read the Phenotype file */

void        iftWriteSelectedTraits(iftPheno *pheno, float *alpha, char *filename); /* Write a value in [0,1] for each trait. 
										    Value 1 means that the trait was selected for grouping, 
										    non-zero values less than 1 means the weight of the trait 
										    in the distance function, and 0 means that the traits was not
										    selected. */

iftDataSet *iftPhenoToDataset(iftPheno *pheno); /* Creates a dataset
					 with all features from the
					 phenotype file to select the
					 best ones. */

void        iftCopyIndivLabels(iftDataSet *Z, iftPheno *pheno); /* Copy the group label of each individual from the dataset to the phenotype data structure */

iftPheno   *iftSelectSubpop(iftPheno *pheno, char *subpop); /* Select a given subpopulation. */

/* -------------------- Methods to manipulate phenotype information -----------*/

iftGeno *iftCreateGeno(int nsnps, int nindividuals, int ngroups); /* Allocate
								     memory
								     to
								     store
								     genotype
								     information. */


void     iftDestroyGeno(iftGeno **geno); /* Deallocate memory with
					       genotype information */
 

iftGeno *iftReadGenotypeInfo(char *filename, int ngroups); /* Read genotype
						      information */


/* --------------- Methods that use both phenotype and genotype information -----------*/

/* Verify if all individuals in the phenotype data are found among the
   individuals in the genotype data and syncronizes these data
   structures. This may imply in deleting individuos from the
   phenotype data which do not have genotype information. The
   individuals must be sorted by group label at the end of this
   process. */

iftPheno *iftSyncronizeData(iftGeno *geno, iftPheno *pheno);



#endif
