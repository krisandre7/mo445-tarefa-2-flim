#ifndef _PLANE_H_
#define _PLANE_H_

#include "common.h"
#include "geometry.h"
#include "context.h"

typedef struct _plane {
  Vector normal;
  Point  po;
  float dx,dy,dz;
  float R[4][4];
  float thx,thy; /* Rotation Angles */
} Plane;

Plane *CreatePlane(Context *cxt);
Plane *CopyPlane(Plane *pl);
void   DestroyPlane(Plane **pl);
void   RotatePlane(Plane *pl, float thx, float thy);
void   RotatePlaneAbs(Plane *pl, float thx, float thy);
void   TranslatePlane(Plane *pl, float alpha);
void   TranslatePlaneAbs(Plane *pl, float alpha);
void   TranslatePlaneAux(Plane *pl, float alpha);
Plane *FacePlane(Vertex *vert, int n);
float  GetPlaneX(Plane *pl);
float  GetPlaneY(Plane *pl);
float  GetPlaneTilt(Plane *pl);
float  GetPlaneSpin(Plane *pl);
void   LockPlane(Context *cxt, Plane *pl);
void   SetPlaneZBuffer(ZBuffer *plane, Plane *pl);
void   SetPlaneShearZBuffer(Context *cxt, ZBuffer *zbuff, Plane *pl);
Plane *ClipPlane(Context *cxt,char axis);
#endif
