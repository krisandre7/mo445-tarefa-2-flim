
#ifndef _GRADIENT_H_
#define _GRADIENT_H_

#include "common.h"
#include "adjacency3.h"
#include "image.h"
#include "cimage.h"
#include "scene.h"
#include "set.h"
#include "radiometric3.h"
#include "morphology3.h"
#include "matrix.h"
#include "geometry.h"
//#include "bz2_lib.h"

#include "preproc.h"
#include "featmap.h"
#include "adjacency_addons.h"

typedef struct _ScnGradient {
  Scene *Gx;
  Scene *Gy;
  Scene *Gz;

  //Available upon request:
  //--> Must call "ComputeScnGradientMagnitude".
  Scene *mag;
} ScnGradient;


#include "mri.h"


ScnGradient *CreateScnGradient(int xsize,int ysize,int zsize);
void         DestroyScnGradient(ScnGradient **grad);
ScnGradient *RemScnGradientFrame(ScnGradient *fgrad, int sz);
ScnGradient *LinearInterpScnGradientCentr(ScnGradient *grad,
					  float dx,float dy,float dz);
ScnGradient *ChangeOrientationToLPS_ScnGradient(ScnGradient *grad,
						char *ori);

ScnGradient *ReadScnGradient(char *filename);
void         WriteScnGradient(ScnGradient *grad, char *filename);

ScnGradient *ReadCompressedScnGradient(char *filename);
void         WriteCompressedScnGradient(ScnGradient *grad, char *filename);

//--------------------------------------

Scene  *LaplacianFilter3D(Scene *orig);

Scene  *SobelFilter3D(Scene *scn);

Scene  *SphericalGradient(Scene *scn, float r);
//--------------------------------------

ScnGradient *SphericalScnGradient(Scene *scn, float r);

void    ComputeScnGradientMagnitude(ScnGradient *grad);
int     ScnGradientMaximumMag(ScnGradient *grad);
void    ScnGradientNormalize(ScnGradient *grad,
			     int omin,int omax,
			     int nmin,int nmax);

void    PowerEnhancementScnGradient(ScnGradient *grad);
void    PowerEnhancementScene(Scene *scn);

//--------------------------------------

ScnGradient *MRI_SphericalScnGradient(Scene *scn, float r, MRI_Info info);
Scene       *MRI_SphericalAccAbsDiff3(Scene *scn, float r, MRI_Info info);


//---------------------------------------------
//---------------------------------------------
//---------------------------------------------
int    vclip(int v,int min,int max);
Scene *TextGradient3(Scene *scn);
Scene *MorphGrad3(Scene *orig, AdjRel3 *A);
Scene *VarianceGrad3(Scene *input, float radius, int thres);
Scene *BrainGrad3(Scene *scn);
Scene *FeatTextGrad3(Scene *scn, FeatMap *fmap);
Scene *FeatGrad3(Scene *scn, FeatMap *fmap);

#endif


