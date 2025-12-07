#ifndef _SCENE_H_
#define _SCENE_H_

#include "image.h"
#include "common.h"
#include "nifti1_io.h"

typedef struct _voxel {
  int x,y,z;
} Voxel;

typedef struct _scene {
  int *data;
  int xsize,ysize,zsize;
  float dx,dy,dz;
  int *tby, *tbz;
  int maxval, n;
  nifti_image *nii_hdr;
} Scene;

void    CopyVoxelSize(Scene *scn1, Scene *scn2);
Scene  *CreateScene(int xsize,int ysize,int zsize);
void    DestroyScene(Scene **scn);
Scene  *CopyScene(Scene *scn);
int     VoxelValue(Scene *scn,Voxel v);
bool    ValidVoxel(Scene *scn, int vx, int vy, int vz);
Voxel   Transform_Voxel(float M[4][4],Voxel v);
int     MaximumValue3(Scene *scn);
int     MinimumValue3(Scene *scn);
int     MaximumValueMask3(Scene *scn, Scene *mask);
int     MinimumValueMask3(Scene *scn, Scene *mask);
Scene  *ROI3(Scene *scn, int xl, int yl, int zl, int xh, int yh, int zh);
Scene  *MBB3(Scene *scn);
Scene  *ReadScene(char *filename);
void    WriteScene(Scene *scn, char *filename);
Image  *GetSlice(Scene *scn, int z);
Image  *GetXSlice(Scene *scn, int x);
Image  *GetYSlice(Scene *scn, int y);
void    PutXSlice(Image *img, Scene *scn, int z);
void    PutYSlice(Image *img, Scene *scn, int z);
void    PutSlice(Image *img, Scene *scn, int z);
Scene  *LinearInterp(Scene *scn,float dx,float dy,float dz);
Scene  *KnnInterp(Scene *scn,float dx,float dy,float dz);
int     KnnInterpValue(Scene *scn, float x, float y, float z);
Scene  *ImageSurface(Image *img);
Scene  *AddFrame3(Scene *scn, int sz, int value);
Scene  *RemFrame3(Scene *fscn, int sz);
Scene  *MergeLabels(Scene *scn1, Scene *scn2);
int     SceneImax(Scene *scn);
void    SetSceneImax(Scene *scn, int val);
Scene  *SubScene(Scene *scn, int x0, int x1, int y0, int y1, int z0, int z1); // Use x1, y1, and/or z1 equal -1 to the maximum size of the scene.

Scene*  Rotate3(Scene *scn,  // Rotate image first around X, then Y and then Z
		double thx, double thy, double thz, // angles to rotate
		float cx, float cy, float cz); // center of the rotation

/* v: index of the voxel, s: size of a slice (xsize*ysize) */
bool EdgeVoxel(Scene *scn, int v, int s);

int GetNSlices(Scene *scn);
int GetXSize(Scene *scn);
int GetYSize(Scene *scn);
int GetZSize(Scene *scn);
float GetDx(Scene *scn);
float GetDy(Scene *scn);
float GetDz(Scene *scn);
void SetDx(Scene *scn, float dx);
void SetDy(Scene *scn, float dy);
void SetDz(Scene *scn, float dz);
void SetVoxelSize(Scene *scn, float dx, float dy, float dz);
int Coord2Voxel(Scene *scn, int x, int y, int z);
int GetVoxelValue(Scene *scn, int voxel);

float GetVoxelValue_trilinear(Scene *scn, float x,float y, float z);
int GetVoxelValue_nn(Scene *scn, float x, float y, float z);

Scene *DrawBorder3(Scene *scn, Scene *label, int value);
int ScenesAreEqual(Scene *scn1, Scene *scn2);
void CompareScenes(Scene *scn1, Scene *scn2);
void SetScene(Scene *scn, int value);
void SetLabelScene(Scene *scn, int label, int value);
Scene *ShapeBasedInterp(Scene *scn,float dx,float dy,float dz);
Scene *Image2Scene(Image *img);
Scene *Image2SceneBin(Image *img);
Scene *ReadScene_Nifti1( char filename[ ] );
void WriteScene_Nifti1( Scene *scn, char filename[ ] );
void CloneNiftiHeader( Scene *src, Scene *dst );
void CopySceneHeader( Scene *src, Scene *dst );

#define GetVoxel(s,n) ((s)->data[(n)])
#define SetVoxel(s,n,i) (s)->data[(n)]=i
#define SceneLen(s) ((s)->xsize * (s)->ysize * (s)->zsize)
#define VoxelX(s,n) (((n) % (((s)->xsize)*((s)->ysize))) % (s)->xsize)
#define VoxelY(s,n) (((n) % (((s)->xsize)*((s)->ysize))) / (s)->xsize)
#define VoxelZ(s,n) ((n) / (((s)->xsize)*((s)->ysize)))
#define VoxelAddress(s,x,y,z) ((x)+(s)->tby[(y)]+(s)->tbz[(z)])

#include "adjacency3.h"

Scene *GetBorder3(Scene *scn, AdjRel3 *A);
void DrawAdj3(Scene *scn, AdjRel3 *A, Voxel *center, int value);
#endif







