
#include "delsamples.h"


namespace Enhancement{

  DelSamples :: DelSamples(ModuleEnhancement *mod){
    this->mod = mod;
  }

  DelSamples :: ~DelSamples(){}


  void DelSamples :: OnEnterWindow(){
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'x');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'y');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'z');
  }

  void DelSamples :: OnLeaveWindow(){}

  void DelSamples :: OnLeftClick(int p){
    int id, o, cod,s;
    Set *S=NULL,*tmp;

    cod = GetVoxel(APP->Data.label, p);

    if(APP->IsSeed(p)){ 
      id = APP->GetSeedId(p);
      o  = APP->GetSeedLabel(p);
    }
    else return;

    if(id == 0) return;

    APP->MarkForRemoval(id);
    S = APP->CopySeeds();
    tmp = S;
    while(tmp!=NULL){
      s = tmp->elem;
      if(APP->IsMarkedForRemoval(s)){
	(APP->Data.label)->data[s] = 0;
      }
      tmp = tmp->next;
    }
    DestroySet(&S);

    APP->DelMarkedForRemoval();
  
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(false, 1.0);
  }


  void DelSamples :: OnRightClick(int p){}

  void DelSamples :: OnMiddleClick(int p){
    mod->Run();
  }

  void DelSamples :: OnLeftDrag(int p, int q){}

  void DelSamples :: OnRightDrag(int p, int q){}

  void DelSamples :: OnMouseMotion(int p){
    APP->Window->SetStatusText(_T("Mouse Left: Select Marker for Removal, Mouse Center: Run"));
  }

} //end Enhancement namespace


