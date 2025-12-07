
#include "modulesurface.h"

namespace Surface{

  ModuleSurface :: ModuleSurface()
    : View3DModule(){
    SetName((char *)"Surface view");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(0,8,0);
    SetVersion(ver);
    panel = NULL;
  }


  ModuleSurface :: ~ModuleSurface(){}


  wxPanel* ModuleSurface :: GetViewPanel(wxWindow *parent){
    SurfaceView *view = new SurfaceView(parent);
    panel = (wxPanel*)view;
    return panel;
  }

  void ModuleSurface :: SetInteractionHandler(InteractionHandler *handler){
    SurfaceView *view;

    view = (SurfaceView *)panel;
    view->canvas->SetInteractionHandler(handler);
  }

  void ModuleSurface :: Start(){
    this->Refresh(true, 1.0);
  }

  bool ModuleSurface :: Stop(){
    return true;
  }

  void ModuleSurface :: Refresh(bool dataChanged, float quality){
    SurfaceView *view;
    SurfaceCanvas *scanvas;
    int skip;

    view = (SurfaceView *)panel;
    scanvas = (SurfaceCanvas *)view->canvas;
    skip = ROUND((1.0-quality)*11)+1;
    scanvas->drawRender(dataChanged, skip);
  }

  void ModuleSurface :: Zoomin(){
    SurfaceView *view;

    view = (SurfaceView *)panel;
    view->canvas->Zoomin();
  }

  void ModuleSurface :: Zoomout(){
    SurfaceView *view;

    view = (SurfaceView *)panel;
    view->canvas->Zoomout();
  }


  void ModuleSurface :: SetZoomLevel(float zoom){
    SurfaceView *view;

    view = (SurfaceView *)panel;
    view->canvas->SetZoomLevel(zoom);
  }

  float ModuleSurface :: GetZoomLevel(){
    SurfaceView *view;
    
    view = (SurfaceView *)panel;
    return view->canvas->GetZoomLevel();
  }


  CImage * ModuleSurface :: CopyAsCImage(){
    SurfaceView *view;
    CImage *cimg;

    view = (SurfaceView *)panel;
    cimg = view->canvas->CopyAsCImage();
    return cimg;
  }


} //end Surface namespace



