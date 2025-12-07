
#include "selection3.h"


BoxSelection3 BoxSelectionByMBB3(Scene *scn){
  BoxSelection3 box;
  int p,i,j,k;
  
  box.v1.x = scn->xsize-1;  box.v2.x = 0;
  box.v1.y = scn->ysize-1;  box.v2.y = 0;
  box.v1.z = scn->zsize-1;  box.v2.z = 0;
  
  for(k=0; k<scn->zsize; k++){
    for(i=0; i<scn->ysize; i++){
      for(j=0; j<scn->xsize; j++){
	p = VoxelAddress(scn,j,i,k);
	if(scn->data[p]!=0){
	  if(j < box.v1.x) box.v1.x = j;
	  if(i < box.v1.y) box.v1.y = i;
	  if(k < box.v1.z) box.v1.z = k;
	  if(j > box.v2.x) box.v2.x = j;
	  if(i > box.v2.y) box.v2.y = i;
	  if(k > box.v2.z) box.v2.z = k;
	}
      }
    }
  }
  return box;
}


BMap  *BoxSelection2BMap3(Scene *scn, BoxSelection3 box){
  BMap *bmap=NULL;
  int p,j,i,k;
  bmap = BMapNew(scn->n);

  box.v1.x = MAX(0, box.v1.x);
  box.v1.y = MAX(0, box.v1.y);
  box.v1.z = MAX(0, box.v1.z);
  box.v2.x = MIN(scn->xsize-1, box.v2.x);
  box.v2.y = MIN(scn->ysize-1, box.v2.y);
  box.v2.z = MIN(scn->zsize-1, box.v2.z);

  for(k=box.v1.z; k<=box.v2.z; k++){
    for(i=box.v1.y; i<=box.v2.y; i++){
      for(j=box.v1.x; j<=box.v2.x; j++){
	p = VoxelAddress(scn,j,i,k);
	_fast_BMapSet1(bmap,p);
      }
    }
  }
  return bmap;
}


int XSizeBoxSel3(BoxSelection3 box){
  return (box.v2.x - box.v1.x + 1);
}


int YSizeBoxSel3(BoxSelection3 box){
  return (box.v2.y - box.v1.y + 1);
}


int ZSizeBoxSel3(BoxSelection3 box){
  return (box.v2.z - box.v1.z + 1);
}


void EnlargeBoxSelection3(BoxSelection3 *box, Scene *scn,
			  int dx, int dy, int dz){
  box->v1.x -= dx;
  box->v2.x += dx; 
  box->v1.y -= dy;
  box->v2.y += dy; 
  box->v1.z -= dz;
  box->v2.z += dz; 
  if(box->v1.x<0)           box->v1.x = 0;
  if(box->v2.x>=scn->xsize) box->v2.x = scn->xsize-1;
  if(box->v1.y<0)           box->v1.y = 0;
  if(box->v2.y>=scn->ysize) box->v2.y = scn->ysize-1;
  if(box->v1.z<0)           box->v1.z = 0;
  if(box->v2.z>=scn->zsize) box->v2.z = scn->zsize-1;
}


Scene *CopySubSceneInBoxSelection3(Scene *scn, 
				   BoxSelection3 box){
  Scene *sub=NULL;
  sub = ROI3(scn,
	     box.v1.x, box.v1.y, box.v1.z,
	     box.v2.x, box.v2.y, box.v2.z);
  SetVoxelSize(sub, scn->dx, scn->dy, scn->dz);
  return sub;
}





