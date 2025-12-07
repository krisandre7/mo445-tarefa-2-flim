
#ifndef _MARKERLIST_H_
#define _MARKERLIST_H_

#include "common.h"
#include "shared.h"
#include "scene_addons.h"


typedef struct _MarkerListNode {
  int p;
  int id;
  int label;
} MarkerListNode;


typedef struct _MarkerList {
  MarkerListNode *data;
  int n;
} MarkerList;


MarkerList *CreateMarkerList(int nseeds);
void        DestroyMarkerList(MarkerList **ML);


//---------------------------------
//---------------------------------
//---------------------------------
#include "shared.h"

void  LoadMarkers3(char *file, Scene *scn, Set **Si, Set **Se);
void  DrawMarkers3(Scene *scn, Set *S, int value);
Set  *Mask2Marker3(Scene *bin, AdjRel3 *A);

float MaxRadiusByErosion3(Scene *bin);
Scene *ObjMaskByErosion3(Scene *bin, float radius);
Scene *BkgMaskByErosion3(Scene *bin, float radius);

/*
Set  *ObjMarkerByErosion3(Scene *bin, float radius);
Set  *BkgMarkerByErosion3(Scene *bin, float radius);

Set  *ObjBandMarker3(Scene *bin, 
		     float inner_radius, 
		     float outer_radius);
Set  *BkgBandMarker3(Scene *bin, 
		     float inner_radius, 
		     float outer_radius);
*/

#endif

