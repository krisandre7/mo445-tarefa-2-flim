#ifndef _FILTERING_H_
#define _FILTERING_H_

#include "adjacency.h"
#include "image.h"
#include "dimage.h"
#include "geometry.h"

typedef struct _kernel {
  float *val;
  AdjRel *adj;
  int xsize,ysize;
} Kernel;

Kernel *MakeKernel(char *coefs);
Kernel *CreateKernel(AdjRel *A);
Kernel *CloneKernel(Kernel *K);
Kernel *NormalizeKernel(Kernel *K);
Kernel *LinearKernel(AdjRel *A);
Kernel *GaussianKernel(AdjRel *A, float stddev); 
Kernel *LaplacianKernel(AdjRel *A, float stddev);
Kernel *GaborKernel(AdjRel *A, float sigma_x, float sigma_y, float theta, float lambda, float psi, float gamma);
void    DestroyKernel(Kernel **K);
void   WriteKernelImage(Kernel *K, char *filename); 
Image  *LinearFilter(Image *img, Kernel *K);
Image  *LinearFilter2(Image *img, Kernel *K);
Image  *LaplacianBorder (Image *img, Kernel *K);
Image  *MedianFilter(Image *img, AdjRel *A);
Image  *ModeFilter(Image *img, AdjRel *A);
Kernel *FoldKernel(Kernel *K);
Image  *SobelFilter(Image *img);
Image  *ImageConvolution(Image *img, Kernel *K);
real   *RealConvolution2D(real *value, Kernel *K, int ncols, int nrows);

/* ----------- IFT-based Operators ------------------- */

Image  *ShapeFilter(Image *bin, float perc);

#endif
