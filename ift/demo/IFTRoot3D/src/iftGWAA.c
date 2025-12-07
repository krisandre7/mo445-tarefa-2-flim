#include "iftRoot3D.h"

/* 
   Author: Alexandre Falcao.  

   Description: This program reads Genotype.csv and a phenotype.csv
   file from a given subpopulation and performs the genome-wide
   association analysis (GWAA). It ouputs SNP, chromosome, position,
   minor allele frequency, and for each group: the number of
   individuals, p-value, first random p-value, the minimum random
   p-value, the maximum random p-value among those that satisfy a
   tolerance percentage, and the relevance of the group. The program
   takes as input parameters: an initial p-value threshold, number of
   random permutations, and a tolerance percentage on the number of
   times that a random p-value can be better than the p-value of the
   group. Pairs (SNP,GROUP), that do not satisfy the initial
   threshold, or the tolerance percentage, are considered irrelevant.

   The p-values are computed as follows. Whenever the minor allele is
   0, then 2's will be the most likely to happen. This will constitute
   the null hypothesis H0 that the number of occurrences k of 2s will
   be higher than the number of occurrences of 0s at a certain
   expected proportion. The pdf for the major allele 2 will be higher
   between n/2 and n than between 0 and n/2. In any case, we should
   accumulate the pdf values greater than or equal to the number k of
   observations, whenever k is above the maximum pdf value (expected
   proportion), and accumulate the pdf values less than or equal to k,
   whenever k is below the maximum pdf value. The result will be the
   p-value --- lower is the p-value higher are the chances of
   rejecting H0 without any mistake. This may happen when there are
   too many 0s (k is low) or when there are too many 2s (k is high)
   much above the expected value. The same reasoning applies for the
   case of 2 being the minor allele. We build the pdf of the major
   allele 0 in this case.

   Currently, it is creating another output file with identification
   "Simpler" in the name (e.g., PValues_SSS_Simpler.csv if the main
   output file is PValues_SSS.csv), which can be used to compute and
   visualize the Manhattan plot using iftGenoPlots and display
   commands. The main idea is that iftGenoPlots should also generate
   the Q-Q plot in the future.
 
*/

/*------------ Data structure used for the QTL analysis --------------------*/
  
typedef struct ift_qtl {
  iftIndiv **indiv;          /* Random sets of grouping solutions */
  int nindivs;               /* Number of individuals in each set */
  int Nsets;                 /* Number of random sets of grouping solutions */
} iftQTL;


/*----------- Methods of this module -------------------------------------- */

iftQTL *iftCreateQTL(iftPheno *pheno, int Nsets); /* Create N
						     selections of
						     random individual
						     groupings */

void iftDestroyQTL(iftQTL *rgroups); /* Destroy random groups in
					memory */


int iftExpectedValue(long double *pdf, int n); /* Find the maximum of a pdf created for n experiments. 
						  That is, the expected number of suceeded experiemts. */

long double iftSignificanceLevel(long double *pdf, int n, int k0, char *pdf_side); /* Compute the significance level for k >= k0, 
										      succeeded experiments, when k0 >= Ek (the expected value),
										      or for k <= k0 succeeded experiments, when k0 < Ek. */


long double *iftNormalDistribution(int n, float alpha); /* Normal distribution with mean and 
							   variance as determined to simulate 
							   a binomial distribution for n experiments with 
							   probability alpha of success in anyone 
							   of these experiments. The normal distribution 
							   is used only if the program detects possible 
							   overflow for factorial computation. */

long double *iftBinomialDistribution(int n, float alpha); /* Binomial distribution for n experiments with 
							     probability alpha of success in anyone 
							     of these experiments. */


void iftFindSignificanceLevels(iftGeno *geno, iftPheno *pheno); /* Compute the significance levels for all pairs (snp,group). 
								   Assuming that the individuals are ordered by groups 
								   from label=1,2,...,ngroups. */

void iftQTLAnalysis(iftGeno *geno, iftPheno *pheno, int Nsets, float tolerance, float threshold); /* For each (snp,group), 
												     it creates sets of random permutations of
												     individuals and verifies the position 
												     of the real p-value with respect to
												     the p-values obtained from those 
												     permutations. The pair (snp,group)
												     is maintained as relevant, whenever the 
												     real p-value is among
												     the X% smallest ones, for a tolerance X. 
												     It assumes that individuals are ordered 
												     by groups from label=1,2,..,ngroups. */
 
void iftWriteSNPSignificLevels(iftGeno *geno, iftPheno *pheno, char *filename); /* Write the output file with the computed p-values */
void iftWriteSNPSignificLevelsSimpler(iftGeno *geno, iftPheno *pheno, char *filename);  /* Write the output file with the computed p-values */



/*-------------- Implementation of the methods -------------------------------*/

iftQTL *iftCreateQTL(iftPheno *pheno, int Nsets)
{
  float  *value=iftAllocFloatArray(pheno->nindivs);
  int    *index=iftAllocIntArray(pheno->nindivs), i, l,g,lo,lf;
  iftQTL *rgroups=(iftQTL *)calloc(1,sizeof(iftQTL));
  
  if (pheno->indiv[0].position == NIL) 
    iftError("You must run iftSyncronizeData first","iftCreateQTL");

  rgroups->indiv = (iftIndiv **)calloc(Nsets,sizeof(iftIndiv *));
  if (rgroups == NULL) 
    iftError(MSG2,"iftCreateQTL");
  for (i=0; i < Nsets; i++) 
    rgroups->indiv[i] = (iftIndiv *) calloc(pheno->nindivs,sizeof(iftIndiv));

  iftRandomSeed(IFT_RANDOM_SEED);
  
  for (i=0; i < Nsets; i++) {
    // initialize sorting vectors 
    for (l=0; l < pheno->nindivs; l++) {
      value[l] = (float)((double) rand () / ((double) RAND_MAX));
      index[l] = l;
    }
    // sort value vector and scramble index vector
    iftFHeapSort(value, index, pheno->nindivs, INCREASING); 
    // copy original individuals from their position in the index vector
    for (l=0; l < pheno->nindivs; l++) {
      strcpy(rgroups->indiv[i][l].lineID,pheno->indiv[index[l]].lineID);
      rgroups->indiv[i][l].label     = pheno->indiv[index[l]].label;
      rgroups->indiv[i][l].position  = pheno->indiv[index[l]].position;
    }
    // relabel the indivividuals to form the new grouping
    lo=0; lf=0;
    for (g=0; g < pheno->ngroups; g++) {
      lf += pheno->nindivs_group[g];
      for (l=lo; l < lf; l++) 
	rgroups->indiv[i][l].label = g+1;
      lo = lf;    
    }
  }

  rgroups->Nsets   = Nsets;
  rgroups->nindivs = pheno->nindivs;

  free(value);
  free(index);

  return(rgroups);
}

void iftDestroyQTL(iftQTL *rgroups)
{
  int i;

  if (rgroups != NULL) {
    for (i=0; i < rgroups->Nsets; i++) 
      free(rgroups->indiv[i]);
    free(rgroups->indiv);
    free(rgroups);
  }    
}


int iftExpectedValue(long double *pdf, int n)
{
  long double Fmax=INFINITY_LDBL_NEG;
  int k,Ek=NIL;
  
  for (k=0; k <= n; k++) 
    if (Fmax < pdf[k]){
      Fmax = pdf[k];
      Ek = k;
    }


  return(Ek);
}


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




void iftFindSignificanceLevels(iftGeno *geno, iftPheno *pheno)
{
  int s,l,g,lo,lf;
  int pos, n0, n2, n, limit=iftFactorialLimit(),N,N0,N2;
  long double *pdf;
  char         pdf_side;

  /* computes p-values for each pair (snp,group) */

#pragma omp parallel for shared(geno,pheno,limit) private(s,l,g,lo,lf,pos,n0,n2,n,N,N0,N2,pdf,pdf_side)
  for (s=0; s < geno->nsnps; s++) {

    /* Compute the minor allele and the minor allele frequency for the snp */

    N0=N2=0;
    for (l=0; l < pheno->nindivs; l++) {
      pos = pheno->indiv[l].position;
      if (geno->snp[s].allele[pos]==0){
	N0++;
      }else{ 
	if (geno->snp[s].allele[pos]==2)
	  N2++;
      }
    }
    
    N = N0+N2;

    if (N > 0) { 

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
	
	lf += pheno->nindivs_group[g]; 
      
	geno->snp[s].pvalue[g]=1.0;
      
	/* compute the number of alleles in {0,2} for this group */
      
	n0=0; n2=0; 
	for (l=lo; l < lf; l++) {     
	  pos  = pheno->indiv[l].position;
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
	      geno->snp[s].pvalue[g] = iftSignificanceLevel(pdf,n,n2,&pdf_side);
	    }else{
	      geno->snp[s].pvalue[g] = iftSignificanceLevel(pdf,n,n0,&pdf_side);
	    }

	    free(pdf);
	    
	  }else{
	    geno->snp[s].group_relevance[g]=0; // eliminate pair (snp,group)
	  }
	}else{
	  geno->snp[s].group_relevance[g]=0; // eliminate pair (snp,group)
	}
	          
	lo = lf;
      }
    } else { // there is no minor allele in {0,2} at snp s
      geno->snp[s].minor_allele      = NIL; // undefined 	
      geno->snp[s].minor_allele_freq = 0;
      for (g=0; g < geno->ngroups; g++) {
	geno->snp[s].group_relevance[g]=0; // eliminate pair (snp,group)
	geno->snp[s].pvalue[g]=1.0;
	geno->snp[s].min_pvalue_perm[g]   =  1.0;
	geno->snp[s].max_pvalue_perm[g]   =  1.0;
	geno->snp[s].first_pvalue_perm[g] =  1.0;
      }
    }
  }

}


void iftQTLAnalysis(iftGeno *geno, iftPheno *pheno, int Nsets, float tolerance, float threshold)
{
  iftQTL      *qtl;
  int          g,l,i,j,s,pos;
  long double *pdf,pvalue,pvalue_aux;
  int          n0, n2, n, limit=iftFactorialLimit();
  int          lo, lf,L=(int)MAX((tolerance*Nsets),1),nlosses;
  long double  pvaluelist[L+1];
  char         pdf_side;
  
  /* generate Nsets random permutations of groups */
  printf("Generating random sets of grouping solutions\n");
  qtl=iftCreateQTL(pheno,Nsets);

  /* compute the QTL analysis */
  printf("Computing the QTL analysis\n");

#pragma omp parallel for shared(geno,pheno,limit,L,qtl,Nsets,tolerance,threshold) private(s,l,i,j,g,lo,lf,pos,n0,n2,n,pdf,pdf_side,pvaluelist,pvalue,pvalue_aux,nlosses)
  for (s=0; s < geno->nsnps; s++) {

    //    printf("SNP %s (%f perc.)\n",geno->snp[s].ID,(float)s/geno->nsnps*100.0);

    if (geno->snp[s].minor_allele != NIL){

      for (g=0, lo=0, lf=0; (g < geno->ngroups); g++, lo = lf) {

	geno->snp[s].min_pvalue_perm[g]   =  1.0;
	for (j=0; j < L; j++) 
	  pvaluelist[j]=INFINITY_LDBL;
	geno->snp[s].first_pvalue_perm[g] =  1.0;
	geno->snp[s].group_relevance[g]   =  1; 
	lf += pheno->nindivs_group[g];

	// apply threshold 

	if (geno->snp[s].pvalue[g] <= threshold) { 

	  for (i=0, nlosses=0; (i < qtl->Nsets)&&(nlosses < L); i++) {

	    /* compute the number of alleles in {0,2} for this group */
	
	    n0=0; n2=0; 
	    for (l=lo; l < lf; l++) {     
	      pos      = qtl->indiv[i][l].position;
	      
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
		  pvalue = iftSignificanceLevel(pdf,n,n2,&pdf_side);
		}else{
		  pvalue = iftSignificanceLevel(pdf,n,n0,&pdf_side);
		}	    

		pvaluelist[L] = pvalue;
		j = L; 
		while ((j>0)&&(pvaluelist[j] < pvaluelist[j-1])){
		  pvalue_aux      = pvaluelist[j-1];
		  pvaluelist[j-1] = pvaluelist[j];
		  pvaluelist[j]   = pvalue_aux; 
		  j--;
		}
		  
		if (pvalue < geno->snp[s].min_pvalue_perm[g]) 
		  geno->snp[s].min_pvalue_perm[g] = pvalue;

		if (geno->snp[s].first_pvalue_perm[g] == 1.0) 
		  geno->snp[s].first_pvalue_perm[g] = pvalue;
	      
		if (pvalue < geno->snp[s].pvalue[g]){
		  nlosses++;
		}
		
		free(pdf);
	      }else{
		geno->snp[s].group_relevance[g]=0; // eliminate (snp,group)
		break;
	      }
	    }else{
	      geno->snp[s].group_relevance[g]=0; // eliminate (snp,group)
	      break;
	    }

	    if (nlosses >= L)
	      geno->snp[s].group_relevance[g]=0; // eliminate (snp,group)
	    
	  }
	}else{
	  geno->snp[s].group_relevance[g]=0; // eliminate (snp,group)
	}

	geno->snp[s].max_pvalue_perm[g]=1.0;
	for (j=L-1; j >= 0; j--)
	  if (pvaluelist[j]<INFINITY_LDBL){
	    geno->snp[s].max_pvalue_perm[g]=pvaluelist[j];
	    break;
	  }
      }
    }
  }

  iftDestroyQTL(qtl);
}

/* void iftQTLAnalysis(iftGeno *geno, iftPheno *pheno, int Nsets, float tolerance, float threshold) */
/* { */
/*   iftQTL      *qtl; */
/*   int          g,l,i,j,s,pos; */
/*   long double *pdf,pvalue,*pvaluelist,pvalue_aux; */
/*   int          n0, n2, n, limit=iftFactorialLimit(); */
/*   int          lo, lf,L=(int)MAX((tolerance*Nsets),1),nlosses; */
/*   char         pdf_side; */
  
/*   /\* generate Nsets random permutations of groups *\/ */
/*   printf("Generating random sets of grouping solutions\n"); */
/*   qtl=iftCreateQTL(pheno,Nsets); */

/*   pvaluelist = iftAllocLongDoubleArray(L+1); */

/*   /\* compute the QTL analysis *\/ */
/*   printf("Computing the QTL analysis\n"); */

/*   for (s=0; s < geno->nsnps; s++) { */

/*     printf("SNP %s (%f perc.)\n",geno->snp[s].ID,(float)s/geno->nsnps*100.0); */

/*     if (geno->snp[s].minor_allele != NIL){ */

/*       for (g=0, lo=0, lf=0; (g < geno->ngroups); g++, lo = lf) { */

/* 	geno->snp[s].min_pvalue_perm[g]   =  1.0; */
/* 	for (j=0; j < L; j++)  */
/* 	  pvaluelist[j]=INFINITY_LDBL; */
/* 	geno->snp[s].first_pvalue_perm[g] =  1.0; */
/* 	geno->snp[s].group_relevance[g]   =  1;  */
/* 	lf += pheno->nindivs_group[g]; */

/* 	// apply threshold  */

/* 	if (geno->snp[s].pvalue[g] <= threshold) {  */

/* 	  for (i=0, nlosses=0; (i < qtl->Nsets)&&(nlosses < L); i++) { */

/* 	    /\* compute the number of alleles in {0,2} for this group *\/ */
	
/* 	    n0=0; n2=0;  */
/* 	    for (l=lo; l < lf; l++) {      */
/* 	      pos      = qtl->indiv[i][l].position; */
	      
/* 	      if (geno->snp[s].allele[pos]==0){ */
/* 		n0++; */
/* 	      }else{ */
/* 		if (geno->snp[s].allele[pos]==2) */
/* 		  n2++; */
/* 	      } */
/* 	    } */
	    
/* 	    n = n0+n2; */
	    
/* 	    /\* generate pdf and compute the p-value of the pair (snp,group) */
/* 	       for allele 2 whenever the minor allele is 0, and for */
/* 	       allele 0, whenever the minor allele is 2. *\/ */
	    
/* 	    if (n > 0) { */
/* 	      if(geno->snp[s].minor_allele_freq > 0.0){ */
/* 		if (n <= limit){ */
/* 		  pdf =  */
/* 		    iftBinomialDistribution(n,1.0-geno->snp[s].minor_allele_freq); */
/* 		}else{ */
/* 		  pdf =  */
/* 		    iftNormalDistribution(n,1.0-geno->snp[s].minor_allele_freq); */
/* 		} */
/* 		if (geno->snp[s].minor_allele==0){ */
/* 		  pvalue = iftSignificanceLevel(pdf,n,n2,&pdf_side); */
/* 		}else{ */
/* 		  pvalue = iftSignificanceLevel(pdf,n,n0,&pdf_side); */
/* 		}	     */

/* 		pvaluelist[L] = pvalue; */
/* 		j = L;  */
/* 		while ((j>0)&&(pvaluelist[j] < pvaluelist[j-1])){ */
/* 		  pvalue_aux      = pvaluelist[j-1]; */
/* 		  pvaluelist[j-1] = pvaluelist[j]; */
/* 		  pvaluelist[j]   = pvalue_aux;  */
/* 		  j--; */
/* 		} */
		  
/* 		if (pvalue < geno->snp[s].min_pvalue_perm[g])  */
/* 		  geno->snp[s].min_pvalue_perm[g] = pvalue; */

/* 		if (geno->snp[s].first_pvalue_perm[g] == 1.0)  */
/* 		  geno->snp[s].first_pvalue_perm[g] = pvalue; */
	      
/* 		if (pvalue < geno->snp[s].pvalue[g]){ */
/* 		  nlosses++; */
/* 		} */
		
/* 		free(pdf); */
/* 	      }else{ */
/* 		geno->snp[s].group_relevance[g]=0; // eliminate (snp,group) */
/* 		break; */
/* 	      } */
/* 	    }else{ */
/* 	      geno->snp[s].group_relevance[g]=0; // eliminate (snp,group) */
/* 	      break; */
/* 	    } */

/* 	    if (nlosses >= L) */
/* 	      geno->snp[s].group_relevance[g]=0; // eliminate (snp,group) */
	    
/* 	  } */
/* 	}else{ */
/* 	  geno->snp[s].group_relevance[g]=0; // eliminate (snp,group) */
/* 	} */

/* 	geno->snp[s].max_pvalue_perm[g]=1.0; */
/* 	for (j=L-1; j >= 0; j--) */
/* 	  if (pvaluelist[j]<INFINITY_LDBL){ */
/* 	    geno->snp[s].max_pvalue_perm[g]=pvaluelist[j]; */
/* 	    break; */
/* 	  } */
/*       } */
/*     }  */
/*   } */
/*   free(pvaluelist); */
/*   iftDestroyQTL(qtl); */
/* } */


void iftWriteSNPSignificLevels(iftGeno *geno, iftPheno *pheno, char *filename)
{
  int   s,g;
  FILE *fp;

  fp = fopen(filename,"w");
  
  fprintf(fp,"ID;Chrom;Pos;MAF");
  for (g=0; g < geno->ngroups; g++) {
    fprintf(fp,";NumofIndiv(%d);p-value(%d);MinRandom-p-value(%d);MaxRandom-p-value(%d);FirstRandom-p-value(%d);Relevance(%d)",g+1,g+1,g+1,g+1,g+1,g+1);
  }
  fprintf(fp,"\n");

  for (s=0; s < geno->nsnps; s++) {

      fprintf(fp,"%s;",geno->snp[s].ID);
      fprintf(fp,"%d;",geno->snp[s].chromosome);
      fprintf(fp,"%d;",geno->snp[s].position);
      fprintf(fp,"%f",geno->snp[s].minor_allele_freq);

      for (g=0; g < geno->ngroups; g++) {

	fprintf(fp,";%d",pheno->nindivs_group[g]);	  
	fprintf(fp,";%Le",(long double)geno->snp[s].pvalue[g]);
	fprintf(fp,";%Le",(long double)geno->snp[s].min_pvalue_perm[g]);
	fprintf(fp,";%Le",(long double)geno->snp[s].max_pvalue_perm[g]);
	fprintf(fp,";%Le",(long double)geno->snp[s].first_pvalue_perm[g]);
	fprintf(fp,";%d",geno->snp[s].group_relevance[g]);
      }

      fprintf(fp,"\n");
  }

  fclose(fp);
}

void iftWriteSNPSignificLevelsSimpler(iftGeno *geno, iftPheno *pheno, char *filename)
{
  int   s,g;
  FILE *fp;
  long double min_pvalue;
  int  group;

  fp = fopen(filename,"w");
  
  fprintf(fp,"ID;Chrom;Pos;p-value;MaxRandom-p-value;FirstRandom-p-value");

  fprintf(fp,"\n");

  for (s=0; s < geno->nsnps; s++) {

      fprintf(fp,"%s;",geno->snp[s].ID);
      fprintf(fp,"%d;",geno->snp[s].chromosome);
      fprintf(fp,"%d;",geno->snp[s].position);

      min_pvalue = INFINITY_LDBL; group = NIL;
      for (g=0; g < geno->ngroups; g++) {
	if (geno->snp[s].group_relevance[g]==1){	  
	  if (geno->snp[s].pvalue[g] < min_pvalue){
	    min_pvalue = geno->snp[s].pvalue[g];
	    group = g;
	  }
	}
      }

      if (group == NIL) {
	fprintf(fp,"%Le;",(long double)1.0);
	fprintf(fp,"%Le;",(long double)1.0);
	fprintf(fp,"%Le",(long double)1.0);
      }else{
	fprintf(fp,"%Le;",(long double)min_pvalue);
	fprintf(fp,"%Le;",(long double)geno->snp[s].max_pvalue_perm[group]);
	fprintf(fp,"%Le",(long double)geno->snp[s].first_pvalue_perm[group]);
      }
      
      fprintf(fp,"\n");
  }

  fclose(fp);
}

int main(int argc, char *argv[]) 
{
  timer          *t1=NULL,*t2=NULL;
  iftPheno   *pheno=NULL;
  iftGeno    *geno=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=6)
    iftError("Usage: iftGWAA <input (phenotype file)> <output (p-values file)> <number of permutations> <tolerance> <initial p-value threshold>","main");

  // Read csv files

  iftRandomSeed(IFT_RANDOM_SEED);

  pheno  = iftReadPhenotypeInfo(argv[1]); 
  iftCompNumIndivsPerGroup(pheno);
  geno   = iftReadGenotypeInfo("Genotype.csv",pheno->ngroups);

  t1     = iftTic();
  printf("Syncronizing genotype and phenotype data\n");
  pheno = iftSyncronizeData(geno,pheno);
  printf("Finding significance levels\n");
  iftFindSignificanceLevels(geno,pheno);
  iftQTLAnalysis(geno,pheno,atoi(argv[3]),atof(argv[4]),atof(argv[5]));
  t2     = iftToc();
  fprintf(stdout,"QTL analysis in %f ms\n",iftCompTime(t1,t2));
  iftWriteSNPSignificLevels(geno, pheno, argv[2]);

  char *token,filename[100];
  token  = strtok(argv[2],".");
  sprintf(filename,"%s_Simpler.csv",token);
  iftWriteSNPSignificLevelsSimpler(geno, pheno, filename);

  iftDestroyPheno(&pheno);
  iftDestroyGeno(&geno);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



