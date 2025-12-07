#include "polynom.h"

Polynom *CreatePolynom(int degree)
{
  Polynom *P=NULL;

  P = (Polynom *) calloc(1,sizeof(Polynom));  
  P->coef   = AllocDoubleArray(degree + 1);
  P->n      = degree;
  return(P);
}

void DestroyPolynom(Polynom **P)
{
  Polynom *aux=NULL;

  aux = *P;
  if(aux != NULL){
    if(aux->coef != NULL) free(aux->coef);
    free(aux);
    *P = NULL;
  }
}

Polynom *DerivPolynom(Polynom *P)
{
  Polynom *D;
  int i,j;

  D = CreatePolynom(P->n-1);
  j = 0;
  for (i=1; i <= P->n; i++){
    D->coef[j] = i*P->coef[i];
    j++;
  }
  return(D);
}

Curve *SamplePolynom(Polynom *P, double from, double to, int nbins)
{
  Curve *curve=NULL;
  double x = from,val;
  double inc = (to-from)/nbins;
  int i,p;
  
  if ((from <= to)&&(nbins > 0)) {
    curve = CreateCurve(nbins);
    for (p=0; p < nbins; p++){
      val=0.0;
      for (i=0; i <= P->n; i++)
	val += pow(x,i)*P->coef[i];
      curve->X[p] = x;
      curve->Y[p] = val;
      x +=inc;
    }
  }
  return(curve);
}

double evaluate(Polynom *P, int x){
  double val;
  int i;

  val=0.0;
  for (i=0; i <= P->n; i++)
    val += pow(x,i)*P->coef[i]; 
  return val;
}

/*
  Curve *SamplePolynom(Polynom *P, double from, double to, double inc)
  {
  Curve *curve=NULL;
  double x,val;
  int i,p,n=(int)(((to-from)/inc)+1);
  
  if (n > 0) {
  curve = CreateCurve(n);
  x=from;
  for (p=0; p < n; p++){
  val=0;
  for (i=0; i <= P->n; i++)
  val += pow(x,i)*P->coef[i];
  curve->X[p] = x;
  curve->Y[p] = val;
  x=x+inc;
  }
  }
  return(curve);
  }*/

Polynom *Regression(Curve *curve, int degree)
{
  Polynom *P=NULL;
  double *A=NULL,*B=NULL;
  int i,j,k;
  double sum, m;

  /* Compute Non-Linear System: A*P=B, where P are the coefficients of
     the polynomial */

  A = AllocDoubleArray((degree+1)*(degree+1));
  B = AllocDoubleArray(degree+1);

  for (i=1; i<= 2*degree; i++){
    sum = 0.0;
    for (k=0; k < curve->n; k++){
      sum += pow(curve->X[k], i);
    }
    if (i<=degree){
      for (j=0; j<=i; j++){ 
	A[(i-j) + j*(degree+1)] = sum;
      }
    }
    else {
      for (j= (i-degree); j<= degree; j++){ 
	A[(i-j) + j*(degree+1)] = sum;
      }
    }
  }
  A[0]= curve->n;
  
  for (i=0; i<=degree; i++){
    sum = 0.0;
    for (k=0; k < curve->n; k++){
      sum += pow(curve->X[k], i)*curve->Y[k];
    }
    B[i] = sum;
  }

  /* Gauss's Regression Method */

  for(k = 0; k < degree; k++){ /* Triangulation of A */
    for(i = k+1; i<= degree; i++){
      m = A[i*(degree+1)+k]/A[k*(degree+1)+k];
      A[i*(degree+1)+k] = 0.0;
      for (j = k+1; j<= degree; j++){
	A[i*(degree+1)+j] = A[i*(degree+1)+j] - m *A[k*(degree+1)+j];
      }
      B[i] = B[i] - m * B[k];
    }
  }
    
  P = CreatePolynom(degree);

  P->coef[degree] = B[degree]/A[degree*(degree+1) + degree];
  for(k=degree-1; k>=0; k--){
    sum= 0.0;
    for(j=k+1; j<=degree; j++){
      sum += A[k*(degree+1)+j]*P->coef[j];
    }
    P->coef[k] = (B[k]-sum)/A[k*(degree+1)+k];
  }
 
  free(A);
  free(B);
  
  return(P);
}


