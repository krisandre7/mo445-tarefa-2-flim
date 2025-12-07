
#ifndef _ADDHANDLERDRIFT_H_
#define _ADDHANDLERDRIFT_H_

#include "startnewmodule.h"
#include "moduledrift.h"

namespace DRIFT{

  class AddHandlerDRIFT : public InteractionHandler {
  public:
    AddHandlerDRIFT(char axis,
	       ModuleDRIFT *mod);
    ~AddHandlerDRIFT();
  
  protected:
    char axis;
    ModuleDRIFT *mod;
    void OnLeftClick(int p);
    void OnRightClick(int p);
    void OnMiddleClick(int p);
    void OnMouseWheel(int p, int rotation, int delta);    
    void OnLeftDrag(int p, int q);
    void OnRightDrag(int p, int q);
    void OnLeftRelease(int p);
    void OnRightRelease(int p);

    void OnMouseMotion(int p);
    void OnEnterWindow();
    void OnLeaveWindow();
  };
} //end DRIFT namespace

#endif

