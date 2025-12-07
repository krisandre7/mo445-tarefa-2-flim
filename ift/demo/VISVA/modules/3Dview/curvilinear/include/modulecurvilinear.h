
#ifndef _MODULECURVILINEAR_H_
#define _MODULECURVILINEAR_H_

#include "startnewmodule.h"
#include "curvilinearview.h"

namespace Curvilinear{

  class ModuleCurvilinear : public View3DModule{
  public:
    ModuleCurvilinear();
    ~ModuleCurvilinear();
    
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
} //end Curvilinear namespace

#endif

