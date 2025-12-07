#ifndef _POLYNOM_H_
#define _POLYNOM_H_

#include "common.h"
#include "curve.h"

typedef struct _polynom { /* Polynomial */
  double *coef; /* a0*x^0 + a1*x^1 + ... + an*x^n */ 
  int n; /* degree n */
} Polynom;

Polynom *CreatePolynom(int degree);
void     DestroyPolynom(Polynom **P);
Curve   *SamplePolynom(Polynom *P, double from, double to, int nbins);
double  evaluate(Polynom *P, int x);
Polynom *Regression(Curve *curve, int degree);
Polynom *DerivPolynom(Polynom *P);

#endif



