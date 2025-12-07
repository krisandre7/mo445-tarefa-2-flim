#include "matrix.h"
#include "common.h"

Matrix *CreateMatrix(int ncols, int nrows)
{
  Matrix *mat=NULL;
  int i;

  mat = (Matrix *) calloc(1,sizeof(Matrix));
  if (mat == NULL){
    Error(MSG1,"CreateMatrix");
  }
  
  mat->val   = AllocDoubleArray(nrows*ncols);
  //mat->val   = (long double *) calloc(1,sizeof(long double));
  mat->tbrow = AllocIntArray(nrows);

  for (i=0; i < nrows; i++)
    mat->tbrow[i]=i*ncols;
  mat->ncols = ncols;
  mat->nrows = nrows;
 
 return(mat);
}

void DestroyMatrix(Matrix **mat)
{
  Matrix *aux;
  
  aux = *mat;
  if(aux != NULL){
    if (aux->val != NULL)   free(aux->val); 
    if (aux->tbrow != NULL) free(aux->tbrow);
    free(aux);    
    *mat = NULL;
  }
}

Matrix *CopyMatrix(Matrix *mat)
{
  Matrix *matc;

  matc = CreateMatrix(mat->ncols,mat->nrows);
  memcpy(matc->val,mat->val,mat->ncols*mat->nrows*sizeof(double));
  
  return(matc);
}

void ShowMatrix(Matrix *M)
{
  int x, y;
  
  printf("\n");
  for (y=0; y<M->nrows; y++){
    for (x=0; x<M->ncols; x++)
      printf("%3.5lf ", M->val[x+M->tbrow[y]]);
    printf("\n");
  }
  printf("\n");
}

void ShowMatrixDimension(Matrix *M)
{  
  printf("\n");
  printf("%d x %d", M->nrows, M->ncols);
  printf("\n");
}

Matrix *Inversion(Matrix *M)
{
  Matrix *A=NULL;
  Matrix *I=NULL;
  int i,j,k;
  double m;
  
  if (M->ncols!=M->nrows){
    printf("Matrix dimension error\n");
    exit(-1);
  }
  
  I = CreateMatrix(M->ncols, M->nrows);
  A = CopyMatrix(M);
  

  for (i=0; i<M->nrows; i++){
    I->val[i + M->tbrow[i]] = 1.0;
  }

 // printf("************* I:\n");
//  ShowMatrix(I);

  for(k = 0; k < M->nrows; k++){
    
    m = A->val[k + M->tbrow[k]];
    if (m < 0.0000000000001){
      printf("ERRO: singular matrix\n");
      exit(-1);
    }
 //   printf("m=%5.10lf\n", m);
    A->val[k + M->tbrow[k]] = 1.0;
    for (j=0; j< M->ncols; j++){
      if (j!=k)
	A->val[j + M->tbrow[k]] = A->val[j + M->tbrow[k]]/m;
      I->val[j + M->tbrow[k]] = I->val[j + M->tbrow[k]]/m;
    }
//    printf("************* I:\n");
//    ShowMatrix(I);
    
    for(i = 0; i< M->nrows; i++){
      if (i!=k){
	m = A->val[k + M->tbrow[i]]/A->val[k + M->tbrow[k]];
//	printf("m=%5.10lf\n", m);
	A->val[k + M->tbrow[i]] = 0.0;
	for (j = 0; j< M->ncols; j++){
	  if (j!=k)
	    A->val[j + M->tbrow[i]] = A->val[j + M->tbrow[i]] - m*A->val[j + M->tbrow[k]];
	  I->val[j + M->tbrow[i]] = I->val[j + M->tbrow[i]] - m*I->val[j + M->tbrow[k]];
	}
      }
 //     printf("************* I:\n");
 //     ShowMatrix(I);
    }
  }
  
  DestroyMatrix(&A);
  return I;
}

double Trace(Matrix *M)
{
  int i;
  double sum;
  
  if (M->ncols!=M->nrows){
    printf("Matrix dimension error\n");
    exit(-1);
  }
  
  sum = 0.0;
  for (i=0; i<M->nrows; i++){
    sum = sum + M->val[i + M->tbrow[i]];
  }
  
  return sum;
}

Matrix *MultMatrix(Matrix *A, Matrix *B)
{
  int i, j, k;
  Matrix *M = NULL;
  
  if (A->ncols!=B->nrows){
    printf("Matrix dimension error\n");
    exit(-1);
  }
  
  M = CreateMatrix(B->ncols, A->nrows);
  for (i=0; i<M->nrows; i++){
    for (j=0; j<M->ncols; j++){
      M->val[j + M->tbrow[i]] = 0.0;
      for (k=0; k<A->ncols; k++){
	M->val[j + M->tbrow[i]] = ((M->val[j + M->tbrow[i]]) +
				   (A->val[k + A->tbrow[i]]*B->val[j + B->tbrow[k]]));
      }
    }
  }
  return(M);
}

Matrix *EscalarMultMatrix(Matrix *A, double k)
{
  int i, j;
  Matrix *M = NULL;
  
  M = CreateMatrix(A->ncols, A->nrows);
  for (i=0; i<M->nrows; i++){
    for (j=0; j<M->ncols; j++){
      M->val[j+M->tbrow[i]] = k*A->val[j+A->tbrow[i]];
    }
  }
  return(M);
}

Matrix *DiffMatrix(Matrix *A, Matrix *B)
{
  int i, j;
  Matrix *M = NULL;
  
  if ((A->ncols!=B->ncols)||(A->nrows!=B->nrows)){
    printf("Matrix dimension error\n");
    exit(-1);
  }
  
  M = CreateMatrix(A->ncols, A->nrows);
  
  for (i=0; i<M->nrows; i++){
    for (j=0; j<M->ncols; j++){
      M->val[j + M->tbrow[i]]=A->val[j + M->tbrow[i]] - B->val[j + M->tbrow[i]];
    }
  }
  return(M);
}

Matrix *SumMatrix(Matrix *A, Matrix *B)
{
  int i, j;
  Matrix *M = NULL;
  
  if ((A->ncols!=B->ncols)||(A->nrows!=B->nrows)){
    printf("Matrix dimension error\n");
    exit(-1);
  }
  
  M = CreateMatrix(A->ncols, A->nrows);
  
  for (i=0; i<M->nrows; i++){
    for (j=0; j<M->ncols; j++){
      M->val[j + M->tbrow[i]]=A->val[j + M->tbrow[i]] + B->val[j + M->tbrow[i]];
    }
  }
  return(M);
}

Matrix *Transpose(Matrix *A)
{
  int i, j;
  Matrix *M = NULL;
    
  M = CreateMatrix(A->nrows, A->ncols);
  
  for (i=0; i<M->nrows; i++){
    for (j=0; j<M->ncols; j++){
      M->val[j + M->tbrow[i]] = A->val[i + A->tbrow[j]];
    }
  }
  return(M);
}

double Matrix_EuclideanDistance(Matrix *Y, Matrix *X)
{
  Matrix *A=NULL, *B = NULL, *R = NULL;
  double r;
  
  A = DiffMatrix(X, Y);
  B = Transpose(A);
  R = MultMatrix(A, B);
  r = sqrt(R->val[0]);  
  DestroyMatrix(&A);
  DestroyMatrix(&B);
  DestroyMatrix(&R);
  return (r);
}
