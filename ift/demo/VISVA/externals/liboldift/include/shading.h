#ifndef _SHADING_H_
#define _SHADING_H_

#include "scene.h"
#include "geometry.h"
#include "context.h"
#include "image.h"
#include "cimage.h"


Image    *DepthShading(Context *cxt);
Image    *PhongShading(Context *cxt);
Image    *PhongShadingScene(Context *cxt,Scene *nscn);
void      ObjectNormal(int Pind, Context *cxt, Vector *normal);
void      SceneNormal(int Vind, Scene *scn, Vector *normal);
CImage   *Colorize(Context *cxt, Image *img);
CImage   *ColorizeSlice(Context *cxt, Image *img);
CImage   *OverlaySlice(Context *cxt, Image *obj, Image *img);
void      AccVoxelColor (CImage *cimg, Context *cxt, int i, float Y, float opac,uchar obj);
void ColorizeCut(Context *cxt, CImage *cimg, Image *img);
CImage *GetXCSlice(Scene *scn, int x);
CImage *GetYCSlice(Scene *scn, int y);
CImage *GetZCSlice(Scene *scn, int z);
Image    *FillSlice(Image *img);
Image    *FillCutPlane(ZBuffer *zbuff, Image *img);
void      FillPlane(CImage *img, Context *cxt);
#endif
