#ifndef _SEEDMAP3_H_
#define _SEEDMAP3_H_

#include "oldift.h"
#include "compressed.h"

#include "filelist.h"
#include "cloud3.h"
#include "fuzzycloud3.h"
#include "floatadjacency3.h"


//Data structures for fast accessing seed pixels of a Cloud.

typedef struct _adjseedmap3 {
  AdjRel3 *disp;  //displacement.
  Cloud3 **uncertainty;
  Cloud3 **object;
  Cloud3 **obj_border;
  Cloud3 **bkg_border;
  int nobjs;
} AdjSeedmap3;


AdjSeedmap3 *CreateAdjSeedmap3(int nobjs);
void         DestroyAdjSeedmap3(AdjSeedmap3 **asmap);

AdjSeedmap3 *RegionCloud2AdjSeedmap3(RegionCloud3 *rcloud);


void DrawAdjSeedmapObject3(Scene *scn,
			   AdjSeedmap3 *asmap,
			   Voxel u,
			   int l,
			   int val);
void DrawAdjSeedmapObjBorder3(Scene *scn,
			      AdjSeedmap3 *asmap,
			      Voxel u,
			      int l,
			      int val);
void DrawAdjSeedmapBkgBorder3(Scene *scn,
			      AdjSeedmap3 *asmap,
			      Voxel u,
			      int l,
			      int val);
void DrawAdjSeedmapUncertainty3(Scene *scn,
				AdjSeedmap3 *asmap,
				Voxel u,
				int l,
				int val);

void CopyAdjSeedmapUncertainty3(Scene *dest,
				Scene *src,
				AdjSeedmap3 *asmap,
				Voxel u,
				int l);

void AddAdjSeedmapUncertainty3(Scene *dest,
			       Scene *src,
			       AdjSeedmap3 *asmap,
			       Voxel u,
			       int l);

void CloudArcWeight3(Scene *arcw,
		     Scene *wobj,
		     ScnGradient *grad,
		     Voxel u,
		     BorderCloud3 *bcloud,
		     AdjSeedmap3 *asmap,
		     int l, float w);

#endif


