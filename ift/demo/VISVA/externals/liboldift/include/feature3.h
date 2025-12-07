#ifndef _FEATURE3_H_
#define _FEATURE3_H_

#include "feature.h"
#include "sort.h"

typedef struct _features3 {
  FElem *elem;
  int  nfeats;
  int  nelems;
  int  xsize,ysize,zsize;
  int  Imax;
} Features3;

Features3* CreateFeatures3(int xsize, int ysize, int zsize, int nfeats);
Features3* CreateSceneFeatures3(Scene *scn, int nfeats);
void      DestroyFeatures3(Features3 **f);
void      DestroySceneFeatures3(Features3 **f);

Features3 *MedianSceneFeats(Scene *scn, Scene *mask, float r);
Features3 *MedianDoubleSceneFeats( Scene *scn1, Scene *scn2, Scene *mask, float r );
Features3 *OCMedianSceneFeats(Scene *scn, Scene *mask, float r, int samples);
Features3 *MarkovSceneFeats(Scene *scn, Scene *mask, float dm);
Features3 *MarkovDoubleSceneFeats( Scene *scn1, Scene *scn2, Scene *mask, float dm );
// features are the voxel bright and nfeats - 1 brights from the closest 1.0 neighbors bright.
Features3 *CoOccur3(Scene *orig, Scene *mask, int nfeats);

#endif // _FEATURE3_H_
