
#ifndef _POSHANDLER_H_
#define _POSHANDLER_H_

#include "startnewmodule.h"
#include "profiledialog.h"

namespace IntensityProfile{

  class PosHandler : public InteractionHandler {
  public:
    PosHandler(ProfileDialog *dialog);
    virtual ~PosHandler();

  protected:
    ProfileDialog *dialog;
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

