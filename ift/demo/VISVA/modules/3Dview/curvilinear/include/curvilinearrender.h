
#ifndef _CURVILINEARRENDER_H_
#define _CURVILINEARRENDER_H_

#include "module.h"

namespace Curvilinear{

  class CurvilinearRender : public SurfaceRender {
  public:
    CurvilinearRender(wxWindow *parent);
    void OnMouseEvent(wxMouseEvent& event);
    //Draw isosurfaces.
    void drawRender(int skip, bool labelChanged);
    
  private:
  };
} //end Curvilinear namespace

#endif


