
#ifndef _LINEHANDLER_H_
#define _LINEHANDLER_H_

#include "startnewmodule.h"
#include "moduleintprofile.h"
#include "profiledialog.h"

namespace IntensityProfile{

  class LineHandler : public InteractionHandler {
  public:
    LineHandler(ModuleIntensityProfile *mod);
    virtual ~LineHandler();

  protected:
    ModuleIntensityProfile *mod;
    Voxel A,B;
    int *line;
    int size,n;
    bool end;
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

