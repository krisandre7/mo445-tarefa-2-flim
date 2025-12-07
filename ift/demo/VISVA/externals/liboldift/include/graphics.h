#ifndef _GRAPHICS_H_
#define _GRAPHICS_H_

#include "image.h"
#include "cimage.h"
#include "scene.h"
#include "annscn.h"
#include "plane.h"
#include "context.h"
#include "adjacency.h"



typedef bool (*FloodFun)(Image *img, int p, int q, int label);

bool FillAnyLabel(Image *img, int p, int q, int label);
bool FillLabel(Image *img, int p, int q, int label);
void ift_FloodFill(Image *img, AdjRel *A, int pos, FloodFun fun, int label);

void      DrawPixel(Image *img, Pixel p, float d, ZBuffer *zbuff);
void      IntensifyPixel (Image *img, int x, int y, double w, float d, ZBuffer *zbuff);
void      DrawLine(Image *img, Pixel p1, Pixel pn, float d1, float dn, ZBuffer *zbuff);
void      DrawAALine(Image *img, Pixel p1, Pixel pn, float d1, float dn, ZBuffer *zbuff);
void      DrawFrame(Image *img, Context *cxt);
void      DrawAAFrame(Image *img, Context *cxt);
void      DrawCutFrame(Image *img, Context *cxt, Plane *pl);
void      DrawPlane(Scene *scn, Context *cxt, Plane *pl, Image *img);
Image    *DrawScene(Scene *scn, Context *cxt);
Image    *DrawCutScene(Scene *scn, Context *cxt, Plane *pl);
Image    *DrawCutObjects(AnnScn *ascn, Context *cxt, Plane *pl);
CImage   *DrawCutObjectsAux(Scene *scn, Scene *label, Context *cxt, Plane *pl);


#endif

