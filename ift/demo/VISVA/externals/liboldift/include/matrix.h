#ifndef _MATRIX_H_
#define _MATRIX_H_

//This module is deprecated.
//This file is still available only for 
//compatibility purposes.
//You should use "RealMatrix" instead,
//which is far more superior. It supports 
//both linear and two-dimensional access
//(M->val[0][p] or M->val[i][j] for a entry
//(i,j) at address p=j+i*ncols) and its
//element values precision can be easily 
//changed (float,double,..).

typedef struct _matrix {
  /*long */double *val;
  int ncols,nrows;
  int *tbrow;
} Matrix;

typedef Matrix* Ap_Matrix;

Matrix  *CreateMatrix(int ncols,int nrows);
void    DestroyMatrix(Matrix **mat);
Matrix  *CopyMatrix(Matrix *mat);
Matrix  *Inversion(Matrix *A);
Matrix  *MultMatrix(Matrix *A, Matrix *B);
Matrix  *EscalarMultMatrix(Matrix *A, double k);
Matrix  *DiffMatrix(Matrix *A, Matrix *B);
Matrix  *SumMatrix(Matrix *A, Matrix *B);
Matrix  *Transpose(Matrix *A);
double  Trace(Matrix *M);
void    ShowMatrix(Matrix *A);
void    ShowMatrixDimension(Matrix *M);
double  Matrix_EuclideanDistance(Matrix *Y, Matrix *X);

#endif



