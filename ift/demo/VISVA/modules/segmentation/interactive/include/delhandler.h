
#ifndef _DELHANDLER_H_
#define _DELHANDLER_H_

#include "startnewmodule.h"
#include "moduleinteractive.h"

namespace Interactive{

  class DelHandler : public InteractionHandler {
  public:
    DelHandler(ModuleInteractive *mod);
    ~DelHandler();

  protected:
    ModuleInteractive *mod;
    void OnLeftClick(int p);
    void OnRightClick(int p);
    void OnMiddleClick(int p);
  
    void OnLeftDrag(int p, int q);
    void OnRightDrag(int p, int q);

    void OnMouseMotion(int p);
    void OnEnterWindow();
    void OnLeaveWindow();
  };
} //end Interactive namespace

#endif

