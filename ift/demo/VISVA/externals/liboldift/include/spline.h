#ifndef _SPLINE_H_
#define _SPLINE_H_

#include "scene.h"
#include "context.h"
#include "plane.h"
#include "image.h"

typedef struct _spline{
  Pixel po[100];
  Voxel vo[100];
  float dx,dy,dz;
  float len[100];
  float total;
  int numpts;
  int max;
} Spline;

Spline *CreateSpline(Scene *scn);
void AddPoints(Spline *spline,int x,int y);
void MovePoint(Spline *spline,int x,int y,int pointnumber);
void DestroySpline(Spline **spline);
void DrawSpline(Scene *scn,Context *cxt,Plane *pl,Spline *spline,Image *img);
void Compute3DSpline(Scene *scn,Context *cxt,Plane *pl,Spline *spline);
void CloseSpline(Spline *spline);
void CalculateVoxel(Spline *spline,float alpha,Vector *v);
Plane *NormalPlane(Scene *scn,Spline *spline,float alpha);
void DrawNormal(Scene *scn,Context *cxt,Plane *pl,Spline *spline,Image *img,float alpha);
float LengthSpline(Spline *spline,float alpha);
Spline *TranslateSpline(Spline *spline,float alpha);
int getCoordx(Spline *spline,int pointnumber);
int getCoordy(Spline *spline, int pointnumber);
#endif
