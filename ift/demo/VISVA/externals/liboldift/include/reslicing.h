#ifndef _RESLICING_H_
#define _RESLICING_H_

#include "scene.h"
#include "context.h"
#include "plane.h"
#include "oldift.h"
#include "spline.h"

Image    *SliceScene(Scene *scn, Context *cxt, Plane *pl);
Image *SliceSceneParallel(Scene *scn,Context *cxt,Plane *pl,Spline *spline);
Scene    *ResliceScene(Scene *scn, Context *cxt, Plane *pl, \
		       float alpha, int nslices);
Scene *ResliceSceneNormal(Scene *scn, Context *cxt, Plane *pl, Spline *spline,float alpha, int nslices);
Scene *ResliceSceneParallel(Scene *scn,Context *cxt,Plane *pl,Spline *spline,float alpha,int nslices);
#endif
