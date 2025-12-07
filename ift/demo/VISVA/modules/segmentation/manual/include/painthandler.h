
#ifndef _PAINTHANDLER_H_
#define _PAINTHANDLER_H_

#include "startnewmodule.h"
#include "modulemanual.h"

namespace Manual{

  class PaintHandler : public InteractionHandler {
  public:
    PaintHandler(char axis,
		 ModuleManual *mod);
    ~PaintHandler();
  
  protected:
    char axis;
    ModuleManual *mod;
    void OnLeftClick(int p);
    void OnRightClick(int p);
    void OnMiddleClick(int p);
    void OnMouseWheel(int p, int rotation, int delta);
    void OnLeftDrag(int p, int q);
    void OnRightDrag(int p, int q);
  
    void OnMouseMotion(int p);
    void OnEnterWindow();
    void OnLeaveWindow();
  };
} //end Manual namespace

#endif

