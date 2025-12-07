
#ifndef _FEATMAP_H_
#define _FEATMAP_H_

#include "common.h"
#include "adjacency3.h"
#include "scene.h"
#include "gqueue.h"
#include "sort.h"
#include "complgraph.h"

#include "shared.h"
#include "scene_addons.h"


typedef struct _FeatMap {
  int n;
  int nfeat;
  real **data;
} FeatMap;


FeatMap *CreateFeatMap(int n, int nfeat);
void     DestroyFeatMap(FeatMap **fmap);

/*Creates a copy of the original "FeatMap".*/
FeatMap *CloneFeatMap(FeatMap *fmap);

FeatMap *Scene2FeatMap(Scene *scn);
FeatMap *Scene2FeatMapByLocalIFT(Scene *scn, Scene *mask, int K);
FeatMap *Scene2FeatMapBy5NumSummary(Scene *scn, Scene *mask, float radius);
FeatMap *Scene2FeatMapByKNN(Scene *scn, Scene *mask, int K, float radius);
FeatMap *Scene2FeatMapByMaxMinVal(Scene *scn, Scene *mask, float radius);

real DistanceSub(real *fv1, real *fv2, int nfeat);
real DistanceL1(real *fv1, real *fv2, int nfeat);
real DistanceL2(real *fv1, real *fv2, int nfeat);
real DistanceGRAD(real *fv1, real *fv2, int nfeat);


//Converte as caracteristicas do volume em um complete 
//graph ordenado pelo label.
ComplGraph *FeatMap2ComplGraph(FeatMap *fmap, Scene *label);

#endif


