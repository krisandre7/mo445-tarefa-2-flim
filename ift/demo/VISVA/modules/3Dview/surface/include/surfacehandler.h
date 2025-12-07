
#ifndef _SURFACEHANDLER_H_
#define _SURFACEHANDLER_H_

#include "startnewmodule.h"
#include "surfacecanvas.h"

namespace Surface{

  class SurfaceHandler : public InteractionHandler {
  public:
    SurfaceHandler(SurfaceCanvas *canvas);
    ~SurfaceHandler();

  protected:
    SurfaceCanvas *canvas;
    void OnLeftClick(int p);
    void OnRightClick(int p);
    void OnMiddleClick(int p);
  
    void OnLeftDrag(int p, int q);
    void OnRightDrag(int p, int q);

    void OnLeftRelease(int p);
    void OnRightRelease(int p);
    void OnMiddleRelease(int p);
  
    void OnMouseMotion(int p);
  };
} //end Surface namespace

#endif

