
#include "modulesetorientation.h"

namespace Setorientation{

  ModuleSetorientation :: ModuleSetorientation()
    : PreProcModule(){
    SetName((char *)"Set Volume Orientation");
    SetAuthor((char *)"Guilherme C. S. Ruppert");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleSetorientation :: ~ModuleSetorientation(){}


  void ModuleSetorientation :: Start(){

    wxWindowDisabler disableAll;
    wxTheApp->Yield();

    SetorientationDialog dia;
    if (dia.ShowModal()==wxID_OK) {
	APP->Busy((char *)"Please wait, working...");
	APP->StatusMessage((char *)"Please wait - Computation in progress...");

	char ori[5];
	Scene *res;
	dia.GetOrientationString(ori);

	//printf("ori: %s\n",ori);

	res = ChangeOrientationToLPS(APP->Data.orig,ori);
	
	SegmObject *obj=NULL;
	int i;
	ArrayList *segmobjs = APP->GetSegmObjsArrayList();
	int n = segmobjs->n;
	for (i=0; i < n; i++) {
	  obj = (SegmObject *)GetArrayListElement(segmobjs, i);
	  Scene *mask1 = CreateScene(APP->Data.orig->xsize,APP->Data.orig->ysize,APP->Data.orig->zsize);
	  CopyBMap2SceneMask(mask1,obj->mask);
	  Scene *mask2 = ChangeOrientationToLPS(mask1,ori);
	  DestroyScene(&mask1);
	  BMap *newbmap;
	  newbmap = SceneMask2BMap(mask2);
	  DestroyScene(&mask2);
	  BMapCopy(obj->mask,newbmap);
	  BMapDestroy(newbmap);
	}

	APP->SetDataVolume_NoDestroy(res);
	APP->Refresh2DCanvas();
	APP->Refresh3DCanvas(true, 1.0);
	APP->DrawSegmObjects();

	APP->Data.oriented=1;

	APP->StatusMessage((char *)"Done.");
	APP->Unbusy();
	APP->Refresh2DCanvas();
	APP->Refresh3DCanvas(true, 1.0);
    }

  }

  bool ModuleSetorientation :: Stop(){
    return true;
  }

  /*
  Scene* ModuleSetorientation :: ChangeOrientationToLPS(Scene *scn, char *ori){
    int Xsrc,Ysrc,Zsrc;
    int xsize,ysize,zsize;
    float dx,dy,dz;
    if (ori[0]=='L') { Xsrc=+1; xsize=scn->xsize; dx=scn->dx; }
    if (ori[0]=='R') { Xsrc=-1; xsize=scn->xsize; dx=scn->dx; }
    if (ori[1]=='L') { Xsrc=+2; xsize=scn->ysize; dx=scn->dy; }
    if (ori[1]=='R') { Xsrc=-2; xsize=scn->ysize; dx=scn->dy; }
    if (ori[2]=='L') { Xsrc=+3; xsize=scn->zsize; dx=scn->dz; }
    if (ori[2]=='R') { Xsrc=-3; xsize=scn->zsize; dx=scn->dz; }

    if (ori[0]=='P') { Ysrc=+1; ysize=scn->xsize; dy=scn->dx; }
    if (ori[0]=='A') { Ysrc=-1; ysize=scn->xsize; dy=scn->dx; }
    if (ori[1]=='P') { Ysrc=+2; ysize=scn->ysize; dy=scn->dy; }
    if (ori[1]=='A') { Ysrc=-2; ysize=scn->ysize; dy=scn->dy; }
    if (ori[2]=='P') { Ysrc=+3; ysize=scn->zsize; dy=scn->dz; }
    if (ori[2]=='A') { Ysrc=-3; ysize=scn->zsize; dy=scn->dz; }

    if (ori[0]=='S') { Zsrc=+1; zsize=scn->xsize; dz=scn->dx; }
    if (ori[0]=='I') { Zsrc=-1; zsize=scn->xsize; dz=scn->dx; }
    if (ori[1]=='S') { Zsrc=+2; zsize=scn->ysize; dz=scn->dy; }
    if (ori[1]=='I') { Zsrc=-2; zsize=scn->ysize; dz=scn->dy; }
    if (ori[2]=='S') { Zsrc=+3; zsize=scn->zsize; dz=scn->dz; }
    if (ori[2]=='I') { Zsrc=-3; zsize=scn->zsize; dz=scn->dz; }


    Scene *res= CreateScene(xsize,ysize,zsize);
    SetVoxelSize(res, dx,dy,dz);
    Voxel v;
    int p,q,oldx,oldy,oldz;
    v.z=0; v.y=0;
    for (v.z=0; v.z < zsize; v.z++)
      for (v.y=0; v.y < ysize; v.y++)
        for (v.x=0; v.x < xsize; v.x++) {
          p = v.x + res->tby[v.y] + res->tbz[v.z];
	  switch (Xsrc) {
	    case +1: oldx=v.x; break;
	    case -1: oldx=xsize-v.x-1; break;
	    case +2: oldy=v.x; break;
	    case -2: oldy=xsize-v.x-1; break;
	    case +3: oldz=v.x; break;
	    case -3: oldz=xsize-v.x-1; break;
	  }
	  switch (Ysrc) {
	    case +1: oldx=v.y; break;
	    case -1: oldx=ysize-v.y-1; break;
	    case +2: oldy=v.y; break;
	    case -2: oldy=ysize-v.y-1; break;
	    case +3: oldz=v.y; break;
	    case -3: oldz=ysize-v.y-1; break;
	  }
	  switch (Zsrc) {
	    case +1: oldx=v.z; break;
	    case -1: oldx=zsize-v.z-1; break;
	    case +2: oldy=v.z; break;
	    case -2: oldy=zsize-v.z-1; break;
	    case +3: oldz=v.z; break;
	    case -3: oldz=zsize-v.z-1; break;
	  }
          q = oldx + scn->tby[oldy] + scn->tbz[oldz];
	  res->data[p]=scn->data[q];
	}

    return res;
  }
  */
  
} //end Setorientation namespace
