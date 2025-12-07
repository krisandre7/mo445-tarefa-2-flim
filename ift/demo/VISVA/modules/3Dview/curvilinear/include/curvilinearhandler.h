
#ifndef _CURVILINEARHANDLER_H_
#define _CURVILINEARHANDLER_H_

#include "startnewmodule.h"
#include "curvilinearcanvas.h"

namespace Curvilinear{

  class CurvilinearHandler : public InteractionHandler {
  public:
    CurvilinearHandler(CurvilinearCanvas *canvas);
    ~CurvilinearHandler();

  protected:
    CurvilinearCanvas *canvas;
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
} //end Curvilinear namespace

#endif

