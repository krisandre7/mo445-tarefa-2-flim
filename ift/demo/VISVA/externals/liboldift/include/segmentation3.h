#ifndef _SEGMENTATION3_H_
#define _SEGMENTATION3_H_

#include "set.h"
#include "adjacency3.h"
#include "scene.h"
#include "annscn.h"
#include "border.h"

Set    *TreeRemoval(AnnScn *ascn, AdjRel3 *A);
void    DIFT(AnnScn *ascn, AdjRel3 *A);
Set    *BTreeRemoval(AnnScn *ascn, AdjRel3 *A, Border *border, BMap *bordermap);
void    BDIFT(AnnScn *ascn, AdjRel3 *A, BMap *isobj, BMap *bordermap, Border *border);
Scene  *Threshold3(Scene *scn, int lower, int higher);
int     AutoThreshold3(Scene *scn);
Scene  *ThresholdMask3(Scene *scn,Scene *mask, int lower, int higher);
Scene  *Highlight3(Scene *scn, Scene *label, int value);
Scene  *LabelComp3(Scene *scn, AdjRel3 *A, int thres);
Scene  *LabelBinComp3(Scene *bin, AdjRel3 *A);
Scene *WaterGray3(Scene *scn, Scene *marker, AdjRel3 *A);
Scene *iftThres3(Scene *scn, AdjRel3 *A);
int Otsu3(Scene *scn);


#endif
