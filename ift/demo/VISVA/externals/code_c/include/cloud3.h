
#ifndef _CLOUD3_H_
#define _CLOUD3_H_

#include "oldift.h"
#include "shared.h"


typedef struct _Cloud3 {
  int *dx;
  int *dy;
  int *dz;
  int n;
  int *dp;

  //private:
  int xsize;
  int xysize;
  int max[3];
  int min[3];
} Cloud3;


Cloud3 *CreateCloud3(int n);
void    DestroyCloud3(Cloud3 **cloud);
Cloud3 *CloneCloud3(Cloud3 *cloud);
Cloud3 *MergeCloud3(Cloud3 *c1, Cloud3 *c2);

Cloud3 *AdjRel2Cloud3(AdjRel3 *A);
Cloud3 *Mask2Cloud3(Scene *mask,
		    Voxel Ref);
Scene  *Cloud32Mask(Cloud3 *cloud);

void    DrawCloud32Scene(Cloud3 *cloud,
			 Scene *scn,
			 Voxel u,
			 int val);
void    DrawOptCloud32Scene(Cloud3 *cloud,
			    Scene *scn,
			    int p, int val);

void    OptimizeCloud3(Cloud3 *cloud,
		       Scene *scn);
void    RefreshCloud3Limits(Cloud3 *cloud);
void    GetCloud3Limits(Cloud3 *cloud,
			int *dx_min, int *dy_min, int *dz_min,
			int *dx_max, int *dy_max, int *dz_max);

int     Cloud3FitInside(Cloud3 *cloud,
			Voxel vx,
			Scene *scn,
			int sz);

float   MeanInsideCloud3(Cloud3 *cloud,
			 Voxel vx,
			 Scene *scn);

float   SumInsideCloud3(Cloud3 *cloud,
			Voxel vx,
			Scene *scn);

float   SumInsideCloudMask3(Cloud3 *cloud,
			    Voxel vx,
			    Scene *scn,
			    Scene *mask);

#endif

