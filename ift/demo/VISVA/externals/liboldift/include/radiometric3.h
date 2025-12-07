#ifndef _RADIOMETRIC3_H_
#define _RADIOMETRIC3_H_

#include "scene.h"
#include "curve.h"

#include "matrix.h"

#include "gqueue.h"
#ifndef _MSC_VER
#include "opfimg.h"
#endif

Curve *Histogram3(Scene *scn);
#ifndef _MSC_VER
Curve *RegionHistogram3( Scene *scn, Subgraph *sg, int label );
Curve *LabelSubGraphHistogram( Scene *scn, LNode *ln );
Curve *LabelSubGraphSimpleHistogram( Scene *scn, LNode *ln, float sampling );
#endif
Curve *NormHistogram3(Scene *scn);
Curve *AccHistogram3(Scene *scn);
Curve *NormAccHistogram3(Scene *scn);
float HistogramMatching( Curve *hp, Curve *hq );
Curve *HistogramMask3(Scene *scn, Scene *mask);
Curve *NormHistogramMask3(Scene *scn, Scene *mask);
Curve *AccHistogramMask3(Scene *scn, Scene *mask);
Curve *NormAccHistogramMask3(Scene *scn, Scene *mask);

Curve *SceneLabels (Scene *scn);
Scene *GaussStretch3(Scene *scn, float mean, float stdev);
Scene *LinearStretch3(Scene *scn, int f1, int f2, int g1, int g2);
Scene *Equalize3(Scene *scn, int Imax);
Scene *EqualizeMask3(Scene *scn, Scene *mask, int Imax);
Scene *MatchHistogram3(Scene *scn, Scene *des);

void   LinearStretchinplace3(Scene *scn,
			     int f1, int f2,
			     int g1, int g2);

Matrix *JoinHistogram(Scene *scn1, Scene *scn2);
Matrix *NormJoinHistogram(Scene *scn1, Scene *scn2);

Scene *MatchHistogramMasks3(Scene *scn, Scene *scn_mask, Scene *des, Scene *des_mask);

Scene *TradEqualize3(Scene *scn);
Scene *MaskEqualize3(Scene *scn,Scene *mask);
Scene *TradMatchHistogram3(Scene *scn, Scene *des);


#endif
