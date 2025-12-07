
#ifndef _METHODS_ADDONS_H_
#define _METHODS_ADDONS_H_


#include "bia_common.h"
#include "bia_adjrel3.h"
#include "bia_scene.h"
#include "bia_set.h"
#include "bia_pqueue16.h"
#include "bia_adjregion3.h"
#include "bia_seedmap3.h"


extern "C" {
#include "common.h"
#include "scene.h"
#include "mri.h"

#include "markerlist.h"
}

//----------------------------------------
//  Method wrappers:

//The 'label' scene should be pre-initialized as follows:
//  label->data[p]=NIL, unlabeled voxel.
//  label->data[p]=0,   background voxel.
//  label->data[p]=1,   object voxel.
/*
void method_IFTSC3(Scene *arcw, Scene *scn,
		   Set *Si, Set *Se,
		   Scene *label);
void method_WeightedDistanceTransform3(Scene *arcw, 
				       Set *Si, Set *Se, 
				       Scene *label,
				       int power);
*/

//----------------------------------------
// Special-purpose optimized code:

typedef struct _IFTSC3AuxiliaryData {
  bia::PQueue16::PQueue16 *Q;
  bia::Scene16::Scene16 *cost;
  bia::AdjRel3::AdjRel3 *A;
  bia::AdjRel3::AdjVxl  *N;
} IFTSC3AuxiliaryData;


IFTSC3AuxiliaryData *CreateIFTSC3AuxiliaryData(bia::Scene16::Scene16 *arcw);
void                 DestroyIFTSC3AuxiliaryData(IFTSC3AuxiliaryData **aux);

real SeedmapIFTSC3(bia::AdjSeedmap3::AdjSeedmap3 *asmap,
		   int index, int lb,
		   bia::Voxel vx,
		   bia::Scene16::Scene16 *arcw,
		   Scene *scn,
		   bia::Scene8::Scene8 *label,
		   MRI_Info info,
		   IFTSC3AuxiliaryData *aux);

/*
real SeedmapWeightedDistance3(AdjSeedmap3 *asmap,
			      int l,
			      Voxel vx,
			      Scene *arcw,
			      Scene *label,
			      int power);
*/

#endif


