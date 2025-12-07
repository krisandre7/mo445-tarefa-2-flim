
#ifndef _ADJACENCY_ADDONS_H_
#define _ADJACENCY_ADDONS_H_

#include "oldift.h"


AdjRel3 *SceneSphericalAdjRel3(Scene *scn, float r);
AdjRel3 *SceneSphericalGridAdjRel3(Scene *scn, float r,
				   float spacement);
void     ScaleAdjRel3(AdjRel3 *A,
		      float Sx, float Sy, float Sz);

void     XClipAdjRel3(AdjRel3 *A, int lower, int higher);
void     YClipAdjRel3(AdjRel3 *A, int lower, int higher);
void     ZClipAdjRel3(AdjRel3 *A, int lower, int higher);

//--------------------------------------
void ClearSceneAdjFrame(Scene *scn, AdjRel3 *A);

void FrameSizes3(AdjRel3 *A, 
		 int *sz_x, 
		 int *sz_y, 
		 int *sz_z);

float   *AdjRel3Distance(AdjRel3 *A);
float   *AdjRel3SceneDistance(Scene *scn, AdjRel3 *A);


#endif

