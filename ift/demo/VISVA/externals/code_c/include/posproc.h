
#ifndef _POSPROC_H_
#define _POSPROC_H_

#include "oldift.h"
#include "shared.h"
#include "scene_addons.h"
#include "adjacency_addons.h"
#include "selection3.h"


void ClearBinBelowThreshold(Scene *scn, Scene *bin, int T);

void ClearPeripheralBinBelowThreshold(Scene *scn, 
				      Scene *bin,
				      float r, int T);

Scene *SmoothObjInterfaces3(Scene *label, float r);

//Scene *SmoothObjInsideMask3(Scene *label, float r,
//	  		      BMap *mask);

void ModeFilterLabel3(Scene *label, float r);


#endif





