#ifndef _RAYCAST_H_
#define _RAYCAST_H_

#include "scene.h"
#include "image.h"
#include "cimage.h"
#include "context.h"

Image    *SWRaycastScene(Scene *scn, Context *cxt);

Image    *RaycastScene(Scene *scn, Context *cxt);
Image    *RaycastGrayScene(Scene *scn, Context *cxt, Scene *nscn);
Image    *BRaycastScene(Scene *scn, Context *cxt);
Image    *RaytracingScene(Scene *scn, Scene *alpha, Scene *normal, Vector *normaltable, Context *cxt);
CImage    *CRaytracingScene(Scene *scn, Scene *alpha, Scene *normal, Vector *normaltable, Context *cxt);
Image    *MIPScene(Scene *scn, Context *cxt);
#endif
