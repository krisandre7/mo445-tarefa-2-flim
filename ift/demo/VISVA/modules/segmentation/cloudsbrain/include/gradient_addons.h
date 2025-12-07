
#ifndef _GRADIENT_ADDONS_H_
#define _GRADIENT_ADDONS_H_

#include "gradient.h"
#include "processors.h"

//Scene       *MRI_SphericalAccAbsDiff3(Scene *scn, float r, MRI_Info info);
//ScnGradient *SphericalScnGradient(Scene *scn, float r);

typedef struct _ArgSphericalScnGradient {
  ScnGradient *grad;
  Scene *scn;
  AdjRel3 *A;
  AdjVxl  *N;
  float  *mg;
  int fx,fy,fz;
  int i; 
  int j;
} ArgSphericalScnGradient;


ScnGradient *FastSphericalScnGradient(Scene *scn, float r);

void *ThreadSphericalScnGradient(void *arg);


//------------------------------------------

typedef struct _ArgMRISphericalAccAbsDiff3 {
  Scene *diff;
  Scene *scn;
  AdjVxl  *N;
  float  *mg;
  float  *weight1,*weight2;
  int dmax;
  int i; 
  int j;
} ArgMRISphericalAccAbsDiff3;


Scene *FastMRISphericalAccAbsDiff3(Scene *scn, float r, MRI_Info info);

void  *ThreadMRISphericalAccAbsDiff3(void *arg);


#endif

