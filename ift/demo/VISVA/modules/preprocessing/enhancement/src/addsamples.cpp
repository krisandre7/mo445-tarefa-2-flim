
#include "addsamples.h"

namespace Enhancement{

  AddSamples :: AddSamples(char axis,
			   ModuleEnhancement *mod){
    this->axis = axis;
    this->mod = mod;
  }

  AddSamples :: ~AddSamples(){}

  void AddSamples :: OnEnterWindow(){
    wxCursor *cursor;
    float zoom = APP->GetZoomLevel();

    cursor = mod->GetBrushCursor(ROUND(zoom));
    APP->Set2DViewCursor(cursor, axis);
  }

  void AddSamples :: OnLeaveWindow(){
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, axis);
  }

  void AddSamples :: OnLeftClick(int p){
    AdjRel *A;
    int cod;

    mod->markerID++;
    cod = mod->GetCodeValue(mod->markerID, 1);
    APP->SetLabelColour(cod, 0xFFFF00);
    A = mod->GetBrush();
    APP->AddSeedsInBrush(p, 1, mod->markerID,
			 A, axis);
    APP->DrawBrush(p, cod, A, axis);

    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void AddSamples :: OnRightClick(int p){
    AdjRel *A;
    int cod;

    mod->markerID++;
    cod = mod->GetCodeValue(mod->markerID, 0);
    APP->SetLabelColour(cod, NIL);
    A = mod->GetBrush();
    APP->AddSeedsInBrush(p, 0, mod->markerID,
			 A, axis);
    APP->DrawBrush(p, cod, A, axis);

    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void AddSamples :: OnMiddleClick(int p){
    mod->Run();
  }


  void AddSamples :: OnMouseWheel(int p, int rotation, int delta){
    if(rotation>0) mod->NextBrush();
    else           mod->PrevBrush();

    wxCursor *cursor;
    float zoom = APP->GetZoomLevel();
    
    cursor = mod->GetBrushCursor(ROUND(zoom));
    APP->Set2DViewCursor(cursor, axis);
  }


  void AddSamples :: OnLeftDrag(int p, int q){
    AdjRel *A;
    int cod;

    cod = mod->GetCodeValue(mod->markerID, 1);
    A = mod->GetBrush();
    APP->AddSeedsInBrushTrace(p,q,1,mod->markerID,
			      A,axis);
    APP->DrawBrushTrace(p,q,cod,A,axis);

    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void AddSamples :: OnRightDrag(int p, int q){
    AdjRel *A;
    int cod;

    cod = mod->GetCodeValue(mod->markerID, 0);
    A = mod->GetBrush();
    APP->AddSeedsInBrushTrace(p,q,0,mod->markerID,
			      A,axis);
    APP->DrawBrushTrace(p,q,cod,A,axis);

    DestroyAdjRel(&A);
    APP->Refresh2DCanvas();
  }

  void AddSamples :: OnMouseMotion(int p){
    wxCursor *cursor=NULL;
    static float zoom = -1.0;
    
    if(zoom!=APP->GetZoomLevel()){
      zoom = APP->GetZoomLevel();
      cursor = mod->GetBrushCursor(ROUND(zoom));
      APP->Set2DViewCursor(cursor, axis);
    }

    APP->Window->SetStatusText(_T("Mouse Left: Add Object Marker, Mouse Center: Run, Mouse Right: Add Background Marker"));
  }

} //end Enhancement namespace

