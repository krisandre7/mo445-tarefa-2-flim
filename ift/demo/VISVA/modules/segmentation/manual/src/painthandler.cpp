
#include "painthandler.h"

namespace Manual{

  PaintHandler :: PaintHandler(char axis,
			       ModuleManual *mod){
    this->axis = axis;
    this->mod = mod;
  }

  PaintHandler :: ~PaintHandler(){}

  void PaintHandler :: OnEnterWindow(){
    wxCursor *cursor;
    float zoom = APP->GetZoomLevel();

    cursor = mod->GetBrushCursor(ROUND(zoom));
    APP->Set2DViewCursor(cursor, axis);
  }

  void PaintHandler :: OnLeaveWindow(){
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, axis);
  }

  void PaintHandler :: OnLeftClick(int p){
    AdjRel *A;
    int o;

    o = mod->obj_sel;
    if(o<0) return;

    A = mod->GetBrush();
    APP->DrawBrush(p, o+1, A, axis);
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void PaintHandler :: OnRightClick(int p){
    AdjRel *A;

    A = mod->GetBrush();
    APP->DrawBrush(p, 0, A, axis);
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void PaintHandler :: OnMiddleClick(int p){
  }

  void PaintHandler :: OnMouseWheel(int p, int rotation, int delta){
    if(rotation>0) mod->NextBrush();
    else           mod->PrevBrush();

    wxCursor *cursor;
    float zoom = APP->GetZoomLevel();
    
    cursor = mod->GetBrushCursor(ROUND(zoom));
    APP->Set2DViewCursor(cursor, axis);
  }

  void PaintHandler :: OnLeftDrag(int p, int q){
    AdjRel *A;
    int o;

    o = mod->obj_sel;
    if(o<0) return;

    A = mod->GetBrush();
    APP->DrawBrushTrace(p,q, o+1, A,axis);
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void PaintHandler :: OnRightDrag(int p, int q){
    AdjRel *A;

    A = mod->GetBrush();
    APP->DrawBrushTrace(p,q, 0,	A,axis);
    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void PaintHandler :: OnMouseMotion(int p){
    wxCursor *cursor=NULL;
    static float zoom = -1.0;
    
    if(zoom!=APP->GetZoomLevel()){
      zoom = APP->GetZoomLevel();
      cursor = mod->GetBrushCursor(ROUND(zoom));
      APP->Set2DViewCursor(cursor, axis);
    }

    APP->Window->SetStatusText(_T("Mouse Left: Paint Object, Mouse Right: Paint Background"));
  }

} //end Manual namespace

