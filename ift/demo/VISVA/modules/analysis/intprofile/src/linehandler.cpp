
#include "linehandler.h"

namespace IntensityProfile{

  LineHandler :: LineHandler(ModuleIntensityProfile *mod){
    this->mod = mod;
    A.x = NIL;
    A.y = NIL;
    A.z = NIL;
    B.x = NIL;
    B.y = NIL;
    B.z = NIL;
    size = 0;
    n = 0;
    line = NULL;
    end = false;
  }

  LineHandler :: ~LineHandler(){
    if(line!=NULL) free(line);
  }

  void LineHandler :: OnEnterWindow(){
    wxCursor *cross=NULL;
    float zoom = APP->GetZoomLevel();
    cross = CrossCursor(ROUND(zoom));
    //wxCROSS_CURSOR
    APP->Set2DViewCursor(cross, 'x');
    APP->Set2DViewCursor(cross, 'y');
    APP->Set2DViewCursor(cross, 'z');
  }


  void LineHandler :: OnLeaveWindow(){
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'x');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'y');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'z');
  }

  void LineHandler :: OnLeftClick(int p){
    Voxel v;
    
    v.x = VoxelX(APP->Data.orig, p);
    v.y = VoxelY(APP->Data.orig, p);
    v.z = VoxelZ(APP->Data.orig, p);
    APP->Set2DViewSlice(v);
    APP->Refresh2DCanvas();
  }
  
  void LineHandler :: OnRightClick(int p){
    ProfileDialog *PDialog=NULL;

    if(A.x==NIL){
      A.x = VoxelX(APP->Data.orig, p);
      A.y = VoxelY(APP->Data.orig, p);
      A.z = VoxelZ(APP->Data.orig, p);
    }
    else if(p!=VoxelAddress(APP->Data.orig,A.x,A.y,A.z)){
      end = true;
      B.x = VoxelX(APP->Data.orig, p);
      B.y = VoxelY(APP->Data.orig, p);
      B.z = VoxelZ(APP->Data.orig, p);

      PDialog = new ProfileDialog(APP->Window, line, this->n, 
				  mod);
      PDialog->ShowModal();
      delete PDialog;
      mod->Stop();
    }
  }
  
  void LineHandler :: OnMiddleClick(int p){}
  
  void LineHandler :: OnLeftDrag(int p, int q) {
    Voxel v;
    v.x = VoxelX(APP->Data.orig, q);
    v.y = VoxelY(APP->Data.orig, q);
    v.z = VoxelZ(APP->Data.orig, q);
    APP->Set2DViewSlice(v);
    APP->Refresh2DCanvas();
  }
  
  void LineHandler :: OnRightDrag(int p, int q){}

  void LineHandler :: OnMouseMotion(int p){
    float d,step,pos;
    Voxel C;
    int i,t,pt;

    APP->Window->SetStatusText(_T("Mouse Left: Navigate, Mouse Right: Select point"));

    if(end==true) return;
    if(A.x==NIL)  return;

    if(line!=NULL){
      for(i=0; i<this->n; i++){
	//APP->DelSeed(line[i]);
	(APP->Data.label)->data[line[i]] = 0;
      }
    }

    B.x = VoxelX(APP->Data.orig, p);
    B.y = VoxelY(APP->Data.orig, p);
    B.z = VoxelZ(APP->Data.orig, p);

    if(p==VoxelAddress(APP->Data.orig,A.x,A.y,A.z)){
      this->n=0;
      APP->Refresh2DCanvas();
      return;
    }

    d = sqrtf((B.x-A.x)*(B.x-A.x) + 
	      (B.y-A.y)*(B.y-A.y) +
	      (B.z-A.z)*(B.z-A.z));

    if(line==NULL){
      size = ROUND(d)+2;
      line = AllocIntArray(size);
    }
    else if(size<ROUND(d)+2){
      size = ROUND(d)+2;
      line = (int *)realloc(line, sizeof(int)*size);
    }



    this->n = 0;
    pt = NIL;
    pos = 0.0;
    step = d/ROUND(d);
    for(i=0; i<ROUND(d)+2; i++){
      C.x = ROUND((A.x*(d-pos) + B.x*pos)/d);
      C.y = ROUND((A.y*(d-pos) + B.y*pos)/d);
      C.z = ROUND((A.z*(d-pos) + B.z*pos)/d);
      t = VoxelAddress(APP->Data.orig,C.x,C.y,C.z);
      if(t!=pt){
	//APP->AddSeed(t, 1, 1);
	(APP->Data.label)->data[t] = 1;
	line[this->n] = t;
	(this->n)++;
	pt = t;
      }
      pos += step;
    }
    APP->Refresh2DCanvas();
  }

} //end IntensityProfile namespace


