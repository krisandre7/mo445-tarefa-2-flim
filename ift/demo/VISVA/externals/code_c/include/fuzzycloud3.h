#ifndef _FUZZYCLOUD3_H_
#define _FUZZYCLOUD3_H_

#include "oldift.h"
#include "compressed.h"

#include "filelist.h"
#include "cloud3.h"
#include "gradient.h"
#include "floatadjacency3.h"
#include "scene_addons.h"


#define MAX_PROB 100000

typedef struct _regionCloud3 {
  AdjRel3  *disp;  //displacement.
  Scene   **prob;
  int nobjs;
  int nimages;

  //private:
  FloatAdjRel3 *fdisp;  //float displacement.
} RegionCloud3;


RegionCloud3 *LabelList2RegionCloud3(FileList *L);
void          DestroyRegionCloud3(RegionCloud3 **rcloud);
void          GetVoxelSizeRegionCloud3(RegionCloud3 *rcloud,
				       float *dx, 
				       float *dy, 
				       float *dz);
RegionCloud3 *SubsamplingRegionCloud3(RegionCloud3 *rcloud);
RegionCloud3 *LinearInterpRegionCloud3(RegionCloud3 *rcloud,
				       float dx,float dy,float dz);
RegionCloud3 *GaussianBlurRegionCloud3(RegionCloud3 *rcloud);
RegionCloud3 *ChangeOrientationToLPSRegionCloud3(RegionCloud3 *rcloud,
						 char *ori);

RegionCloud3 *ReadRegionCloud3(char *filename);
void          WriteRegionCloud3(RegionCloud3 *rcloud, 
				char *filename);

void RemoveElemRegionCloud3(RegionCloud3 *rcloud,
			    Scene *label);

//---------------------------------------------

typedef struct _borderCloud3 {
  AdjRel3      *disp;  //displacement.
  ScnGradient **prob;
  int nobjs;

  //private:
  FloatAdjRel3 *fdisp;  //float displacement.
} BorderCloud3;


BorderCloud3 *RegionCloud2BorderCloud3(RegionCloud3 *rcloud);

void          DestroyBorderCloud3(BorderCloud3 **bcloud);
void          GetVoxelSizeBorderCloud3(BorderCloud3 *bcloud,
				       float *dx, 
				       float *dy, 
				       float *dz);
BorderCloud3 *SubsamplingBorderCloud3(BorderCloud3 *bcloud);
BorderCloud3 *LinearInterpBorderCloud3(BorderCloud3 *bcloud,
				       float dx,float dy,float dz);
BorderCloud3 *ChangeOrientationToLPSBorderCloud3(BorderCloud3 *bcloud,
						 char *ori);
void          NormalizeBorderCloud3(BorderCloud3 *bcloud,
				    int omin,int omax,
				    int nmin,int nmax);

BorderCloud3 *ReadBorderCloud3(char *filename);
void          WriteBorderCloud3(BorderCloud3 *bcloud, 
				char *filename);

#endif


