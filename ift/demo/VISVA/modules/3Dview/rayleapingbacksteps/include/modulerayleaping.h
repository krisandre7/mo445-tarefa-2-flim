
#ifndef _MODULERAYLEAPING_H_
#define _MODULERAYLEAPING_H_

#include "startnewmodule.h"
#include "rayleapingview.h"

namespace Rayleaping{

  class ModuleRayleaping : public View3DModule{
  public:
    ModuleRayleaping();
    ~ModuleRayleaping();

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
} //end Rayleaping namespace

#endif

