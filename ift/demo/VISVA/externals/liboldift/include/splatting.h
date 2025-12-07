#ifndef _SPLATTING_H_
#define _SPLATTING_H_


#include "scene.h"
#include "context.h"
#include "border.h"
#include "image.h"


Image *SplatScene(Scene *scn, Context *cxt);
Image *SWSplatScene(Scene *scn, Context *cxt);
Image *SplatBorder(Scene *scn, Context *cxt, Border *border);

#endif
