#ifndef _FILTERING3_H_
#define _FILTERING3_H_

#include "adjacency3.h"
#include "scene.h"
#include "geometry.h"

typedef struct _kernel3 {
  float *val;
  AdjRel3 *adj;
  int xsize,ysize,zsize;
} Kernel3;

Kernel3 *MakeKernel3(char *coefs);
Kernel3 *CreateKernel3(AdjRel3 *A);
Kernel3 *CloneKernel3(Kernel3 *K);
Kernel3 *NormalizeKernel3(Kernel3 *K);
Kernel3 *FoldKernel3(Kernel3 *K);
Kernel3 *LinearKernel3(AdjRel3 *A);
Kernel3 *GaussianKernel3(AdjRel3 *A, float stddev);
Kernel3 *LaplacianKernel3(AdjRel3 *A, float stddev);
void     DestroyKernel3(Kernel3 **K);
Scene   *LinearFilter3(Scene *scn, Kernel3 *K);
Scene   *SobelFilter3(Scene *scn);
Scene   *MedianFilter3(Scene *scn, AdjRel3 *A);
Scene    *ModeFilter3(Scene *scn, AdjRel3 *A);
real    *RealConvolution3D(real *value, Kernel3 *K, int xsize,int ysize,int zsize);

/* ----------- IFT-based Operators ------------------- */



#endif
