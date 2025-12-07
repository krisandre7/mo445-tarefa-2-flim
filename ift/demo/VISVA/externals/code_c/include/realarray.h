// version 00.00.02

#ifndef _REALARRAY_H_
#define _REALARRAY_H_

#include "oldift.h"

real  MaxValInRealArray(real *A, int n);
real  MinValInRealArray(real *A, int n);
void  RealArrayStatistics(real *A, int n,
			  real *mean,
			  real *stdev);
int  *RealArray2IntArray(real *A, int n,
			 int Imin, int Imax);
real *ReadTxt2RealArray(char *filename, int *n);

#endif

