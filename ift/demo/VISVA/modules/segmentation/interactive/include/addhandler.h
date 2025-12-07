
#ifndef _ADDHANDLER_H_
#define _ADDHANDLER_H_

#include "startnewmodule.h"
#include "moduleinteractive.h"

namespace Interactive{

  class AddHandler : public InteractionHandler {
  public:
    AddHandler(char axis,
	       ModuleInteractive *mod);
    ~AddHandler();
  
  protected:
    char axis;
    ModuleInteractive *mod;
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
} //end Interactive namespace

#endif

