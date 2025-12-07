#include "ift.h"

/*
   Author: Alexandre Falcao. 

   Description:

   For a given number of trials, his program creates a statistical
   table (StatTable.csv) with significance levels for the binomial
   distribution upto overflow in factorial computation, and completes
   the table with the significance levels for the normal
   distribution. In both cases, the probability alpha of the expected
   value is 0.5 to create the null hypotesis H0 (we use one-tail
   analysis, but we will consider two alternative hypotheses H1 and H2
   around H0).

   The null hypothesis H0 is that alpha = 0.5, so the expected result
   of an experiment may be related to either H1 or H2 with equal
   probabilities, and any observed value k0 is irrelevant. The
   alternative hypotheses H1 and H2 are that alpha = k0/n for n trials
   and the observed value k may be k >= k0, when k0 is above the point
   Kc of maximum density value in the binomial distribution, or k <=
   k0, when k0 is below that point. In order to accept H1 or H2, we
   need to obtain a significance level p very low by assuming the
   probability distribution of H0 (where alpha=0.5). If p < 0.05, for
   example, then we accept H1 and reject H0 for k0 >= kc, and the same
   is valid when we accept H2 and reject H0 for k0 < kc.

*/

/* Find the point of maximum pdf value */

int iftExpectedValue(long double *pdf, int n)
{
  long double Fmax=INFINITY_LDBL_NEG;
  int k,kc=NIL;
  
  for (k=0; k <= n; k++) 
    if (Fmax < pdf[k]){
      Fmax = pdf[k];
      kc = k;
    }
  return(kc);
}

/* Compute the significance level for k >= k0, when k0 >= kc (the
   expected value), or for k <= k0, when k0 < kc. */

long double iftSignificanceLevel(long double *pdf, int n, int k0)
{
  long double p=0.0; 
  int k, kc=iftExpectedValue(pdf,n);

  
  if (k0 >= kc){
    for (k=k0; k <= n; k++) 
      p += pdf[k];
  }else{
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

int main(int argc, char *argv[]) 
{
  timer  *t1=NULL,*t2=NULL;
  int     k, n, max_n, limit;
  long double *pdf=NULL;
  FILE   *fp=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=2)
    iftError("Usage: iftStatTable <number of trials>","main");

  max_n     = atoi(argv[1]); // maximum number of trials 
 
  t1     = iftTic();

  fp = fopen("StatTable.csv","w");

  limit = iftFactorialLimit();
  fprintf(fp,"%d,%d,\n",max_n,limit);

  if (max_n <= limit) 
    limit = max_n;

  for (n=1; n <= limit; n++) {    
    pdf = iftBinomialDistribution(n,0.5);
    for (k=0; k <= n; k++){ 
      fprintf(fp,"%Le,",(long double)iftSignificanceLevel(pdf,n,k));
    }
    fprintf(fp,"\n");
    free(pdf);
  }
  for (n=limit+1; n <= max_n; n++) {    
    pdf = iftNormalDistribution(n,0.5);
    for (k=0; k <= n; k++){ 
      fprintf(fp,"%Le,",(long double)iftSignificanceLevel(pdf,n,k));
    }
    fprintf(fp,"\n");
    free(pdf);
  }

  fclose(fp);

  t2     = iftToc();
  fprintf(stdout,"Significance levels computed in %f ms\n",iftCompTime(t1,t2));

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDynamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



