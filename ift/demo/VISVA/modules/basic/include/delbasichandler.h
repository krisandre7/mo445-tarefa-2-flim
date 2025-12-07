
#ifndef _DELBASICHANDLER_H_
#define _DELBASICHANDLER_H_

#include "startnewmodule.h"

namespace Basic{

  class DelHandler : public InteractionHandler {
  public:
    DelHandler();
    virtual ~DelHandler();

  protected:
    virtual void OnLeftClick(int p);
    virtual void OnRightClick(int p);
    virtual void OnMiddleClick(int p);
    
    virtual void OnLeftDrag(int p, int q);
    virtual void OnRightDrag(int p, int q);
    
    virtual void OnMouseMotion(int p);
  };
}

#endif

