
#ifndef _NAVIGATORBASICHANDLER_H_
#define _NAVIGATORBASICHANDLER_H_

#include "startnewmodule.h"

namespace Basic{

  class NavigatorHandler : public InteractionHandler {
  public:
    NavigatorHandler();
    virtual ~NavigatorHandler();

  protected:
    virtual void OnLeftClick(int p);
    virtual void OnRightClick(int p);
    virtual void OnMiddleClick(int p);
  
    virtual void OnLeftDrag(int p, int q);
    virtual void OnRightDrag(int p, int q);
  
    virtual void OnMouseMotion(int p);
    virtual void OnEnterWindow();
    virtual void OnLeaveWindow();
  };
}

#endif

