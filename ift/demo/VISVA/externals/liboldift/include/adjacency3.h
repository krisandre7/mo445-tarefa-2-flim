#ifndef _ADJACENCY3_H_
#define _ADJACENCY3_H_

#include "common.h"


typedef struct _adjrel3 {
  int *dx;
  int *dy;
  int *dz;
  int n;
} AdjRel3;

typedef struct _adjvxl {
  int *dp;
  int n;
} AdjVxl;

AdjRel3 *CreateAdjRel3(int n);

void     DestroyAdjRel3(AdjRel3 **A);
AdjRel3 *CloneAdjRel3(AdjRel3 *A);

AdjRel3 *Spheric(float r);
AdjRel3 *Ellipsoid( float rx, float ry, float rz );
AdjRel3 *SphericalShell(float inner_radius,
			float outer_radius);

int      FrameSize3(AdjRel3 *A);

AdjRel3 *Cube(int xsize, int ysize, int zsize);
void     DestroyAdjVxl(AdjVxl **N);

#include "scene.h"
AdjVxl  *AdjVoxels(Scene *scn, AdjRel3 *A);

int ValidAdjacentAddress( Scene *scn, AdjRel3 *A, int p, int i );
int ValidAdjacentAddressMask( Scene *mask, AdjRel3 *A, int p, int i );


#endif
