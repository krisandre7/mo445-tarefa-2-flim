#ifndef _SHELL_H_
#define _SHELL_H_

#include "scene.h"
#include "image.h"
#include "cimage.h"
#include "context.h"
#include "curve.h"
#include "geometry.h"
#include "plane.h"

typedef struct _attrib {
  uchar obj;
  uchar opac;
  uchar visible;
  ushort x;
  ushort normal;
} Attrib;

typedef struct _cattrib {
  ushort normalmag;
  int val;  
} CAttrib;


typedef struct _pattrib {
  ushort y;
  int i;
} PAttrib;


typedef struct _shell {
  Attrib *voxel;
  CAttrib *body;
  PAttrib *pointer;
  int xsize,ysize,zsize;
  int nvoxels;
  float dx,dy,dz;
  Image *Myz;
  Image *Mzx;
  Vector *normaltable;
  int maxval;
  uchar nobjs;
  Scene *scn;
} Shell;



/* alpha is opacity scene from 0-255 */
Shell  *CreateShell (Scene *scn, Scene *alpha, int w); 
Shell  *CreateBodyShell (Scene *scn, Scene *alpha);
Shell  *NewShell (int xsize, int ysize, int zsize, int nvoxels);
Shell  *NewBodyShell (int xsize, int ysize, int zsize, int nvoxels);
Shell  *Object2Shell (Scene *scn,int w);
Shell  *Scene2Shell (Scene *scn); /* Create body shell with voxels > 0 */
Shell  *MergeShell(Shell *shell1, Shell *shell2);
Shell  *ReadShell (char *filename);
Shell  *ReadBodyShell (char *filename);


Attrib  *ReadVoxels (FILE *fp, int n);
PAttrib *ReadList (FILE *fp, int n);
CAttrib *ReadBody (FILE *fp, int n);
Image   *ReadM(FILE *fp, int ncols, int nrows);
int      VoxelExist (Shell *shell, Voxel *v);

void DestroyShell (Shell **shell);
void WriteShell (Shell *shell, char *filename);
void WriteBodyShell (Shell *shell, char *filename);
void WriteVoxels (FILE *fp, Attrib *voxel, int n);
void WriteList (FILE *fp, PAttrib *list, int n);
void WriteBody (FILE *fp, CAttrib *list, int n);
void WriteM(FILE *fp, Image *img);

/*shell->scene*/

Scene *CreateAlpha (Scene *scn);
Scene *InnerBorder (Scene *bin, int w);
Scene *ObjectAlpha (Scene *scn, int w);
Scene *OpacityScene (Shell *shell);
Scene *NormalIndexScene (Shell *shell);
Scene *LabelScene (Shell *shell);
Scene *FilledLabelScene (Shell *shell);
Scene *ObjectScene (Shell *shell, int obj);
Scene *ValueScene (Shell *shell);
Scene *RemoveVoxels (Shell *shell, float low, float hi);
Scene *NonTransparentVoxels (Scene *scn, Scene *alpha);

/* shell slice -> image*/

Image *GetBodyShellXSlice (Shell *shell, int x);
Image *GetBodyShellYSlice (Shell *shell, int y);
Image *GetBodyShellZSlice (Shell *shell, int z);
Image *GetShellXSlice (Shell *shell, int x);
Image *GetShellYSlice (Shell *shell, int y);
Image *GetShellZSlice (Shell *shell, int z);

/* shell statistics */

Curve *ShellObjHistogram(Shell *shell);
Curve *ShellHistogram(Shell *shell);
Curve *ShellNormHistogram(Shell *shell);
Curve *ShellGradientHistogram(Shell *shell);


/*utility set functions*/

void SetShellVoxels (Shell *shell, Scene *scn);
void SetShellNormalIndex (Shell *shell, Scene *nscn);
void SetShellList (Shell *shell, Scene *scn);
void SetBodyShellValue (Shell *shell, Scene *scn);
void SetShellLabel (Shell *shell, Scene *scn);
void SetShellOpacity (Shell *shell, Scene *alpha);
void SetObjectVisibility (Shell *shell, int obj, int visible);
void SetBodyShellVisibility (Shell *shell, int lower, int higher);

/*utility get functions*/

int GetShellXSize (Shell *shell);
int GetShellYSize (Shell *shell);
int GetShellZSize (Shell *shell);
int GetShellMaximum (Shell *shell);
int GetShellMinimum (Shell *shell);
int GetPointer (Shell *shell, Voxel *v);
int GetShellNObjects(Shell *shell);


/*gray rendering*/

Image   *SWShellRendering  (Shell *shell,Context *cxt);
Image   *ShellRendering (Shell *shell,Context *cxt);
Image   *MIPShell(Shell *shell, Context *cxt);

/*color rendering*/

CImage  *CShellRendering (Shell *shell,Context *cxt);
CImage  *CSWShellRendering(Shell *shell,Context *cxt);
CImage  *CMIPShell(Shell *shell, Context *cxt);
CImage  *CShellRenderingStereo(Shell *shell,Context *cxt);








#endif








