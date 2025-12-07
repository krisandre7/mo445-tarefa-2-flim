#ifndef _REALMATRIX_H_
#define _REALMATRIX_H_

#include "common.h"
#include "image.h"

typedef struct _realmatrix {
  real **val;
  int ncols,nrows;
} RealMatrix;


RealMatrix *CreateRealMatrix(int ncols,int nrows);
void        DestroyRealMatrix(RealMatrix **mat);
RealMatrix *CloneRealMatrix(RealMatrix *mat);
void        CopyRealMatrix(RealMatrix *dest, 
			   RealMatrix *src);

RealMatrix *InvertRealMatrix(RealMatrix *A);
RealMatrix *TransposeRealMatrix(RealMatrix *A);

RealMatrix *MultRealMatrix(RealMatrix *A, 
			   RealMatrix *B);
RealMatrix *MultRealMatrixByScalar(RealMatrix *A, 
				   real k);

RealMatrix *DiffRealMatrix(RealMatrix *A, 
			   RealMatrix *B);
RealMatrix *SumRealMatrix(RealMatrix *A, 
			  RealMatrix *B);

real        CompRealMatrixTrace(RealMatrix *M);

void        PrintRealMatrix(RealMatrix *M);
void        PrintRealMatrixDimension(RealMatrix *M);

real        RealMatrixDistanceL2(RealMatrix *Y, 
				 RealMatrix *X);

void        SetRealMatrix(RealMatrix *M, 
			  real value);
void        ChangeValueRealMatrix(RealMatrix *M, 
				  real old_value,
				  real new_value);

bool        ValidRealMatrixEntry(RealMatrix *M,
				 int i, int j);

Image      *ConvertRealMatrix2Image(RealMatrix *M);

RealMatrix *ReadRealMatrix(char *filename);
void        WriteRealMatrix(RealMatrix *M,
			    char *filename);

real        MinimumRealMatrixValue(RealMatrix *M);
real        MaximumRealMatrixValue(RealMatrix *M);

Image      *ThresholdRealMatrix(RealMatrix *M,
				real lower, 
				real higher);

RealMatrix* RotationMatrix3(int axis, // options: 0 (x) / 1 (y) / 2 (z)
			    double th); 

RealMatrix* TranslationMatrix3(float dx, float dy, float dz);
RealMatrix* ScaleMatrix3(float Sx, float Sy, float Sz);
RealMatrix* ShearMatrix3(float SHxy,float SHxz,float SHyx,float SHyz,float SHzx,float SHzy);


RealMatrix* TransformVoxel(RealMatrix *m, Voxel v);



#endif


