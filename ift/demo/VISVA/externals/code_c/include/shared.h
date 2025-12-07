
#ifndef _SHARED_H_
#define _SHARED_H_

#include "oldift.h"
#include "filelist.h"


void UpdateLabelFromMask3(Scene *label, int l, Scene *mask);
void ClearLabelOutsideMask3(Scene *label, int l, Scene *mask);

void DrawMask3(Scene *scn, Scene *mask, int val);

void PrintDots(int n);

real CompSetMean(Set *S);


void WriteVoxelArray(Voxel *v, int n,
		     char *filename);
void ReadVoxelArray(Voxel *v, int *n,
		    char *filename);


void    DrawSet2Scene(Scene *scn, Set *S, 
		      int value);
Set    *Mask2Set3(Scene *mask);



int   IntegerNormalize(int value,
		       int omin,int omax,
		       int nmin,int nmax);

int           GetMarkerID(int mk);
unsigned char GetMarkerLabel(int mk);
int           GetMarkerValue(int id, unsigned char lb);

BMap  *SceneLabel2BMap(Scene *label);
BMap  *SceneMask2BMap(Scene *mask);
BMap  *SceneMarker2BMap(Scene *marker);

void   CopyBMap2SceneMask(Scene *mask, BMap *bmap);
Scene *SceneMarker2Label(Scene *marker);

Set   *SceneBorder(Scene *pred);


Scene *MyLabelBinComp3(Scene *bin, AdjRel3 *A);

void   SelectLargestComp(Scene *ero);

Scene *Mask2EDT3(Scene *bin, AdjRel3 *A, char side, int limit, char sign);
Scene *SceneMask2EDT3(Scene *bin, AdjRel3 *A,
		      char side, int limit, char sign);


void          BIA_RemoveDirectory(char *dest, char *src);

void          StartTimer(timer *t);
void          StopTimer(timer *t);


void   MyRemoveDirectory(char *dest, char *src);

//-----------------------------------
//-----------------------------------
//-----------------------------------


Scene *EraseBackground(Scene *scn, Scene *mask, Set *obj, Set *bkg);
Scene *WatershedMask3(Scene *scn, Scene *mask, Set *obj, Set *bkg);

void    ConvertCImageList2jpeg(FileList *L);
void    ConvertCImageList2ppm(FileList *L);

CImage *MergeCImages(CImage *A, CImage *B);

void    MergeCImageList(FileList *in1,
			FileList *in2,
			FileList *out);

CImage *CopySubCImage(CImage *cimg, 
		      int j1, int i1, 
		      int j2, int i2);

#endif

