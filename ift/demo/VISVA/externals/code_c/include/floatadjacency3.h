
#ifndef _FLOATADJACENCY3_H_
#define _FLOATADJACENCY3_H_

#include "common.h"


typedef struct _floatadjrel3 {
  float *dx;
  float *dy;
  float *dz;
  int n;
} FloatAdjRel3;

FloatAdjRel3 *CreateFloatAdjRel3(int n);

void          DestroyFloatAdjRel3(FloatAdjRel3 **A);
FloatAdjRel3 *CloneFloatAdjRel3(FloatAdjRel3 *A);

FloatAdjRel3 *ChangeOrientationToLPSFloatAdjRel3(FloatAdjRel3 *A,
						 char *ori);

#endif

