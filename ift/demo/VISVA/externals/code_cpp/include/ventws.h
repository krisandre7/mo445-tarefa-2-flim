

#ifndef _VENTWS_H_
#define _VENTWS_H_

extern "C" {
#include "oldift.h"
#include "simetria.h"
}

/* typedef struct _nplane { */
/*   DVoxel normal; */
/*   DVoxel center; */
/*   double xshift; */
/* } NPlane; */

typedef struct _comp {
  int left;
  int right;
  int val;
} Comp;

typedef struct _elem {
  int first;
  int last;
} Elems;

typedef struct _hist {
  int val;
  int nitems;
} Hist;


typedef int (*pt2Func)(Hist *, Hist *);

int AscSort(  void *,   void *);

int NumericSort(  void *,   void *);

int DescSort(  void * x,   void * y);

pt2Func MySort(char opCode);

int EucDist3D(int x, int y, int z, DVoxel v);

void FreeHist(Hist **hist);

int MyPlaneInfo(const char *filename, PlaneInfo *plane);
  
Elems Parse(char *arg);

Hist * ClosestToCenter(Scene *bin, PlaneInfo *plane, AdjRel3 *A);

Scene * LargestSceneComp(Scene *comp, int ncomps);

void RemoveCenterVoxels(Scene *bin, PlaneInfo *plane);

void RemoveBorderVoxels(Scene *bin, Scene *border);

void MySelectLargestComp(Scene *ero);

float MyDistance(float *f1, float *f2, int n);

Scene *TextGradient(Scene *scn);

int MyOtsu(Scene *scn);

void MeansAboveBelowT(Scene *scn, int T, int *T1, int *T2);

Scene *ApplySShape(Scene *scn, int a, int b, int c);

Scene *enhance(Scene *in, int otsu);

Scene * GradientScene(Scene *in, int e, int otsu);

Scene * BrainMarkerScene(Scene *scn, int otsu);

Scene * VentricleMarkerScene(Scene *scn, Elems elems, double adj,
                             int erode, PlaneInfo plane, int otsu);

Scene * StandardDeviation(Scene *scn, Scene *bin);

Scene * GetVentricle (const char *basename, Scene *iscn,
                      double adj, int erode, int debug);

#endif
