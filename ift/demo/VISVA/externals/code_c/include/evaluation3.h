// version 00.00.03

#ifndef _EVALUATION3_H_
#define _EVALUATION3_H_

#include "oldift.h"
#include "shared.h"
#include "scene_addons.h"

real    DiceSimilarity3(Scene *mask1,
			Scene *mask2);
real    JaccardSimilarity3(Scene *mask1,
			   Scene *mask2);
real    CentrDiceSimilarity3(Scene *mask1,
			     Scene *mask2);
real    CentrJaccardSimilarity3(Scene *mask1,
				Scene *mask2);

real    CentrMObjDiceSimilarity3(Scene *label1,
				 Scene *label2);

//mask1: Ground Truth
//mask2: Segmentation Result
int     AssessTP3(Scene *mask1, Scene *mask2);
int     AssessFN3(Scene *mask1, Scene *mask2);
int     AssessFP3(Scene *mask1, Scene *mask2);
int     AssessTN3(Scene *mask1, Scene *mask2);

#endif

