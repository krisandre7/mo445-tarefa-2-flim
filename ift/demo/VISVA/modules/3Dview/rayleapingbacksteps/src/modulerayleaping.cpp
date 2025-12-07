
#include "modulerayleaping.h"

namespace Rayleaping{

  ModuleRayleaping :: ModuleRayleaping()
    : View3DModule(){
    SetName((char *)"Surface view (ray leaping)");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(0,8,0);
    SetVersion(ver);
    panel = NULL;
  }


  ModuleRayleaping :: ~ModuleRayleaping(){}


  wxPanel* ModuleRayleaping :: GetViewPanel(wxWindow *parent){
    RayleapingView *view = new RayleapingView(parent);
    panel = (wxPanel*)view;
    return panel;
  }

  void ModuleRayleaping :: SetInteractionHandler(InteractionHandler *handler){
    RayleapingView *view;

    view = (RayleapingView *)panel;
    view->canvas->SetInteractionHandler(handler);
  }

  void ModuleRayleaping :: Start(){
    this->Refresh(true, 1.0);
  }

  bool ModuleRayleaping :: Stop(){
    return true;
  }

  void ModuleRayleaping :: Refresh(bool dataChanged, float quality){
    RayleapingView *view;
    RayleapingCanvas *scanvas;
    int skip;

    view = (RayleapingView *)panel;
    scanvas = (RayleapingCanvas *)view->canvas;
    skip = ROUND((1.0-quality)*11)+1;
    scanvas->drawRender(skip);
  }

  void ModuleRayleaping :: Zoomin(){
    RayleapingView *view;

    view = (RayleapingView *)panel;
    view->canvas->Zoomin();
  }

  void ModuleRayleaping :: Zoomout(){
    RayleapingView *view;

    view = (RayleapingView *)panel;
    view->canvas->Zoomout();
  }


  void ModuleRayleaping :: SetZoomLevel(float zoom){
    RayleapingView *view;

    view = (RayleapingView *)panel;
    view->canvas->SetZoomLevel(zoom);
  }

  float ModuleRayleaping :: GetZoomLevel(){
    RayleapingView *view;
    
    view = (RayleapingView *)panel;
    return view->canvas->GetZoomLevel();
  }


  CImage * ModuleRayleaping :: CopyAsCImage(){
    RayleapingView *view;
    CImage *cimg;

    view = (RayleapingView *)panel;
    cimg = view->canvas->CopyAsCImage();
    return cimg;
  }


} //end Rayleaping namespace



