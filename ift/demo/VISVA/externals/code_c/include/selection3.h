
#ifndef _SELECTION3_H_
#define _SELECTION3_H_

#include "oldift.h"
#include "shared.h"


typedef struct _boxSelection3 {
  Voxel v1;
  Voxel v2;
} BoxSelection3;


BoxSelection3 BoxSelectionByMBB3(Scene *scn);
BMap  *BoxSelection2BMap3(Scene *scn, BoxSelection3 box);

int XSizeBoxSel3(BoxSelection3 box);
int YSizeBoxSel3(BoxSelection3 box);
int ZSizeBoxSel3(BoxSelection3 box);

void EnlargeBoxSelection3(BoxSelection3 *box, Scene *scn,
			  int dx, int dy, int dz);

Scene *CopySubSceneInBoxSelection3(Scene *scn, 
				   BoxSelection3 box);


#endif
