#ifndef _MORPHOLOGY3_H_
#define _MORPHOLOGY3_H_

#include "scene.h"
#include "annscn.h"
#include "adjacency3.h"
#include "mathematics3.h" 

Scene *Dilate3(Scene *scn, AdjRel3 *A);
Scene *Erode3(Scene *scn, AdjRel3 *A);
Scene *Open3(Scene *scn, AdjRel3 *A);
Scene *Close3(Scene *scn, AdjRel3 *A);
Scene *MorphGrad3(Scene *scn, AdjRel3 *A);
Scene *SupRec3(Scene *scn, Scene *marker, AdjRel3 *A);
Scene *InfRec3(Scene *scn, Scene *marker, AdjRel3 *A);
Scene *CloseHoles3(Scene *scn);
Scene *RemDomes3(Scene *scn);
Scene *OpenRec3(Scene *scn, AdjRel3 *A);
Scene *CloseRec3(Scene *scn, AdjRel3 *A);
Scene *Leveling3(Scene *scn1, Scene *scn2);
void iftBasins3(AnnScn *ascn, AdjRel3 *A);
void iftDomes3(AnnScn *ascn, AdjRel3 *A);
Scene *AreaClose3(Scene *scn, int thres);
Scene *AreaOpen3(Scene *scn, int thres);
Scene *DilateBin3(Scene *bin, Set **seed, float radius);
Scene *ErodeBin3(Scene *bin, Set **seed, float radius);
Scene *CloseBin3(Scene *bin, float radius);
Scene *OpenBin3(Scene *bin, float radius);
Scene *CloseBinRec3(Scene *bin, float radius);
Scene *OpenBinRec3(Scene *bin, float radius);
Scene *AsfOCBin3(Scene *bin, float radius);
Scene *AsfCOBin3(Scene *bin, float radius);
Scene *AsfOCBinRec3(Scene *bin, float radius);
Scene *AsfCOBinRec3(Scene *bin, float radius);

Scene *AsfOCRec3(Scene *scn, AdjRel3 *A);
Scene *AsfCORec3(Scene *scn, AdjRel3 *A);
#endif
