
#ifndef _SEGMOBJECT_H_
#define _SEGMOBJECT_H_

#include "common.h"
#include "scene.h"
#include "set.h"

#include "markerlist.h"

typedef struct _SegmObject {
  int   color;
  int   alpha;
  int   visibility;
  char  name[1024];
  BMap       *mask;
  MarkerList *seed;

} SegmObject;


SegmObject *CreateSegmObject(char *name,
			     int color);
void        DestroySegmObject(SegmObject **obj);

void        CopySegmObjectMask2Scene(SegmObject *obj, Scene *dest);

void        freeSegmObject(void **obj);

#endif

