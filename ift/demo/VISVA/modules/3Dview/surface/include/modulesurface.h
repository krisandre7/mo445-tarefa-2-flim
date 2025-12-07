
#ifndef _MODULESURFACE_H_
#define _MODULESURFACE_H_

#include "startnewmodule.h"
#include "surfaceview.h"

namespace Surface{

  class ModuleSurface : public View3DModule{
  public:
    ModuleSurface();
    ~ModuleSurface();

    wxPanel *GetViewPanel(wxWindow *parent);
    void  SetInteractionHandler(InteractionHandler *handler);
    void  Start();
    bool  Stop();
    void  Refresh(bool dataChanged, float quality);
    void  Zoomin();
    void  Zoomout();
    void  SetZoomLevel(float zoom);
    float GetZoomLevel();
    CImage *CopyAsCImage();
    
  protected:
    wxPanel *panel;
  };
} //end Surface namespace

#endif

