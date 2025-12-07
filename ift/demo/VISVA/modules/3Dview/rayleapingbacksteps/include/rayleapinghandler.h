
#ifndef _RAYLEAPINGHANDLER_H_
#define _RAYLEAPINGHANDLER_H_

#include "startnewmodule.h"
#include "rayleapingcanvas.h"

namespace Rayleaping{

  class RayleapingHandler : public InteractionHandler {
  public:
    RayleapingHandler(RayleapingCanvas *canvas);
    ~RayleapingHandler();

  protected:
    RayleapingCanvas *canvas;
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
} //end Rayleaping namespace

#endif

