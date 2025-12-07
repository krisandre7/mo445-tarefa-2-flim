
#ifndef _PREPROC_H_
#define _PREPROC_H_

#include "oldift.h"
#include "shared.h"
#include "scene_addons.h"
#include "radiometric_addons.h"
#include "processors.h"


void   SuppressHighIntensities(Scene *scn);

int    SceneMeanInMask(Scene *scn, Scene *mask);
int    MeanAboveThreshold(Scene *scn, int T);
int    SceneRangeMeanValue(Scene *scn, int lower, int higher);

void   MeansAboveBelowT(Scene *scn, int T, int *T1, int *T2);
Scene *ApplySShape(Scene *scn, int a, int b, int c);

int    ComputeOtsu3(Scene *scn);

int    ComputeOtsuMask3(Scene *scn, Scene *mask);

/* Compute mean and stdev of the input scene */
void   DescriptiveStatisticsMask3(Scene *scn,
				  Scene *mask,
				  real *mean,
				  real *stdev);

void   OrderStatisticMask3(Scene *scn,
			   Scene *mask,
			   real *median);

Scene *Convolution3(Scene *scn, Kernel3 *K);
Scene *OptConvolution3(Scene *scn, Kernel3 *K);

Kernel3 *SphericalGaussianKernel3(float R, float s, float f);

Scene   *GaussianBlur3(Scene *scn);
Scene   *OptGaussianBlur3(Scene *scn);
Scene   *FastGaussianBlur3(Scene *scn);
Scene   *FastOptGaussianBlur3(Scene *scn);

Scene   *Subsampling3(Scene *scn);

//--------------------------------
//--------------------------------
//--------------------------------

Scene *MeanLocalIFT3D(Scene *scn, Scene *mask, int rsize);
Scene *MedianLocalIFT(Scene *scn, Scene *mask, int rsize);
Scene *MeanLocalIFT(Scene *scn, Scene *mask, int rsize);
Scene *WeightMeanLocalIFT(Scene *scn, Scene *mask, int rsize);


Scene *InfRecMarker(Scene *iscn, Scene *marker);

//--------------------------------

typedef struct _ArgConvolution3 {
  Kernel3 *K;
  AdjVxl *vxl;
  Scene *scn;
  Scene *cscn;
  int i; 
  int j;
} ArgConvolution3;


Scene *FastConvolution3(Scene *scn, Kernel3 *K);
void  *ThreadConvolution3(void *arg);

Scene *FastOptConvolution3(Scene *scn, Kernel3 *K);
void  *ThreadOptConvolution3(void *arg);


#endif

