
#include "modulecurvilinear.h"

namespace Curvilinear{

  ModuleCurvilinear :: ModuleCurvilinear()
    : View3DModule(){
    SetName((char *)"Curvilinear view");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(0,8,0);
    SetVersion(ver);
    panel = NULL;
  }

  ModuleCurvilinear :: ~ModuleCurvilinear(){}

  wxPanel* ModuleCurvilinear :: GetViewPanel(wxWindow *parent){
    CurvilinearView *view = new CurvilinearView(parent);
    panel = (wxPanel*)view;
    return panel;
  }

  void ModuleCurvilinear :: SetInteractionHandler(InteractionHandler *handler){
    CurvilinearView *view;

    view = (CurvilinearView *)panel;
    view->canvas->SetInteractionHandler(handler);
  }

  void ModuleCurvilinear :: Start(){
    this->Refresh(true, 1.0);
  }

  bool ModuleCurvilinear :: Stop(){
    return true;
  }

  void ModuleCurvilinear :: Refresh(bool dataChanged, float quality){
    CurvilinearView *view;

    view = (CurvilinearView *)panel;
    view->CurvilinearRefresh(dataChanged, quality);
  }

  void ModuleCurvilinear :: Zoomin(){
    CurvilinearView *view;

    view = (CurvilinearView *)panel;
    view->canvas->Zoomin();
  }

  void ModuleCurvilinear :: Zoomout(){
    CurvilinearView *view;

    view = (CurvilinearView *)panel;
    view->canvas->Zoomout();
  }


  void ModuleCurvilinear :: SetZoomLevel(float zoom){
    CurvilinearView *view;

    view = (CurvilinearView *)panel;
    view->canvas->SetZoomLevel(zoom);
  }

  float ModuleCurvilinear :: GetZoomLevel(){
    CurvilinearView *view;
    
    view = (CurvilinearView *)panel;
    return view->canvas->GetZoomLevel();
  }


  CImage * ModuleCurvilinear :: CopyAsCImage(){
    CurvilinearView *view;
    CurvilinearCanvas *ccanvas;
    CImage *cimg;

    view = (CurvilinearView *)panel;
    ccanvas = (CurvilinearCanvas *)view->canvas;
    cimg = ccanvas->CopyAsCImage();
    return cimg;
  }


} //end Curvilinear namespace

