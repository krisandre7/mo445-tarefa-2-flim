
#include "delbasichandler.h"

namespace Basic{

  DelHandler :: DelHandler(){}

  DelHandler :: ~DelHandler(){}

  void DelHandler :: OnLeftClick(int p){
    int id;

    if(!APP->IsSeed(p)) return;
    id = APP->GetSeedId(p);
    if(id == 0) return;

    if(APP->IsIdMarkedForRemoval(id))
      APP->UnmarkForRemoval(id);
    else
      APP->MarkForRemoval(id);
  
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(false, 1.0);
  }


  void DelHandler :: OnRightClick(int p){}

  void DelHandler :: OnMiddleClick(int p){
    //Reset:
    /*
    APP->ResetData();

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    */
  }

  void DelHandler :: OnLeftDrag(int p, int q){}

  void DelHandler :: OnRightDrag(int p, int q){}

  void DelHandler :: OnMouseMotion(int p){
    APP->Window->SetStatusText(_T("Mouse Left: Select Marker for Removal"));
  }

} //end Basic namespace

