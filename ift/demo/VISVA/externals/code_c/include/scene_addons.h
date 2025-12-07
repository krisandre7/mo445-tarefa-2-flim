
#ifndef _SCENE_ADDONS_H_
#define _SCENE_ADDONS_H_

#include "oldift.h"
#include "shared.h"
#include "processors.h"

Scene *GetSceneTransitions3(Scene *scn, AdjRel3 *A);

//Returns the distance in millimeters.
float VoxelEuclideanSquaredDistance(Scene *scn,
				    Voxel u, Voxel v);


Scene *ComputeIntegralScene(Scene *scn);

//The dimensions of the window (i.e., xsize,ysize,zsize) 
//should be given in millimeters.
Voxel  FindWindowOfMaximumDensity(Scene *scn, 
				  float xsize, 
				  float ysize, 
				  float zsize);

Scene *ChangeOrientationToLPS(Scene *scn, char *ori);

Scene *LinearInterpCentr3(Scene *scn,float dx,float dy,float dz);

Scene *LinearRotate3(Scene *scn, 
		     double thx, double thy, double thz, // angles to rotate
		     int cx, int cy, int cz); // center of the rotation

Scene *ShapeBasedRotate3(Scene *scn,
			 double thx, double thy, double thz,
			 int cx, int cy, int cz);


void    SumMaskCentr3(Scene *acc,
		      Scene *mask);
void    ComputeMaskCentroid3(Scene *mask, 
			     Voxel *C);


void    ComputeSceneMBB(Scene *scn,
			Voxel *Vm,
			Voxel *VM);
Scene  *CopySubScene(Scene *scn,
		     Voxel Vmin,
		     Voxel Vmax);
void    PasteSubScene(Scene *scn,
		      Scene *sub,
		      Voxel pos);

Scene  *AddSceneFrame(Scene *scn, int sx, int sy, int sz, int value);
Scene  *RemSceneFrame(Scene *fscn, int sx, int sy, int sz);


Scene *WeightedMean3(Scene *scn1, Scene *scn2, float w,
		     int min1, int max1, 
		     int min2, int max2,
		     int nmin, int nmax);

void   WeightedMean3inplace(Scene *scn1, Scene *scn2, 
			    float w,
			    int min1, int max1, 
			    int min2, int max2,
			    int nmin, int nmax);


void  SceneNormalize(Scene *scn,
		     int omin,int omax,
		     int nmin,int nmax);

void   SetSceneFrame(Scene *scn, int sz, int value);
void   CopySceneFrame(Scene *dest, Scene *src, int sz);

Scene   *InterpScene2Isotropic(Scene *scn);

Scene *BIA_InterpScene2Isotropic(Scene *scn, ArrayList *segmobjs);

Scene *BIA_LinearInterp(Scene *scn,float dx,float dy,float dz, ArrayList *segmobjs);

int           GetRadiometricRes3(Scene *scn);


/* Compute mean and stdev of the input scene */
void          ComputeDescriptiveStatistics(Scene *scn, 
					   float *mean, 
					   float *stdev);


float *SceneIntensity2Gaussian(Scene *scn, int mean,
			       float stdev_lower, 
			       float stdev_higher);

//------------------------------------

typedef struct _ArgLinearInterpCentr3 {
  Scene *scn;
  Scene *iscn;
  int i; 
  int j;
} ArgLinearInterpCentr3;


Scene *FastLinearInterpCentr3(Scene *scn,float dx,float dy,float dz);

void  *ThreadLinearInterpXCentr3(void *arg);
void  *ThreadLinearInterpYCentr3(void *arg);
void  *ThreadLinearInterpZCentr3(void *arg);


#endif

