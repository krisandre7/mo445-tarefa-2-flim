#ifndef _CONTEXT_H_
#define _CONTEXT_H_

#include "frame.h"
#include "algebra.h"
#include "scene.h"
#include "adjacency.h"
#include "filtering.h"

typedef struct _zbuffer{
  float *dist;
  uchar *object;
  int   *voxel;
  int usize,vsize;
  int *tbv;
} ZBuffer;

typedef struct _fbuffer{
  float *val;
  int usize,vsize;
  int *tbv;
} FBuffer;

typedef struct _phobject{
  float Y,Cb,Cr;
  float opac;
  float diff,spec,amb; /* Phong's Model */
  int   ns; /* Phong's Model */
  ushort xoffset;
  ushort yoffset;
  ushort zoffset;
} Object;

typedef struct _context {
  float thx,thy; /* Rotation Angles */
  int   xsize,ysize,zsize;
  Frame  *fr;
  float IR[4][4],R[4][4]; /* Inverse and Rotation Matrices */
  ZBuffer *zbuff; /* Z-buffer */
  float IW[4][4],W[4][4]; /* Inverted Warp and Warp Matrices. The
                             third line and column are never used in
                             2D. */
  float *Su,*Sv,*ISu,*ISv; /* Look-up tables for shear and inverted
			      shear transformations */
  char  PAxis; /* 'x','y', or 'z' */
  char  oct; /* 1,2,...,8 */
  int   xi,xf,dx,yi,yf,dy,zi,zf,dz;
  int   xmin,xmax,ymin,ymax,zmin,zmax; /* Clipping parameters */
  int   ki,kf,dk;
  Object obj[256];
  Vector viewer;
  Vector view_up;
  Vector xaxis,yaxis,zaxis;

  int width,height,depth;

  /* SW specific */
  float Si,Sj,Ti,Tj;
  float isize,jsize,ksize;

  Kernel *footprint;
  AdjPxl *fprint;
  float *fopac;
  
} Context;

Context  *CreateContext(Scene *scn);
Context  *NewContext(int xsize, int ysize, int zsize);
void      DestroyContext(Context **cxt);
void      SetAngles(Context *cxt, float theta_x, float theta_y);
void      AddAngles (Context *cxt, float thx, float thy);
void      SetPhong(Context *cxt, uchar oindex, float diff, float spec,\
		    float amb, int ns);
void      SetObjectColor(Context *cxt, uchar oindex, float red, float green, \
			 float blue);
void      SetObjectOpacity(Context *cxt, uchar oindex, float opac);
void      SetClip(Context *cxt, int xmin, int xmax, int ymin, int ymax, \
		  int zmin, int zmax);
void      SetAxisIndices(Context *cxt);
ZBuffer  *CreateZBuffer(int usize, int vsize);
void      DestroyZBuffer(ZBuffer **zbuff);
void      InitZBuffer(ZBuffer *zbuff);
void      InitZBufferObject(ZBuffer *zbuff, uchar obj);
void      InitZBufferVoxel(ZBuffer *zbuff);

FBuffer *CreateFBuffer(int usize, int vsize);
void     DestroyFBuffer(FBuffer **buff);
void     InitFBuffer(FBuffer *fbuff, float f);

float GetTilt (Context *cxt);
float GetSpin (Context *cxt);

int GetZBufferVoxel (ZBuffer *zbuff, int i);
float GetZBufferDist (ZBuffer *zbuff, int i);
uchar GetZBufferObject (ZBuffer *zbuff, int i);
void  SetZBufferDist(ZBuffer *zbuff,FBuffer *dist);

#endif
