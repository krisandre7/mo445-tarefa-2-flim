#ifndef _FRAME_H_
#define _FRAME_H_

#include "geometry.h"

typedef struct _frame {
  int xsize, ysize, zsize;
  Quad   face[6];
  Edge   edge[12];
  Vertex vert[8];
} Frame; 

Frame    *CreateFrame(int xsize, int ysize, int zsize);
void      DestroyFrame(Frame **fr);
void      SetFrameVert(Frame *fr, int xmin, int xmax, int ymin, int ymax, \
		      int zmin, int zmax);

#endif
