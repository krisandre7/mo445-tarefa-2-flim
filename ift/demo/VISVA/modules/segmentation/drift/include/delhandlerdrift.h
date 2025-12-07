
#ifndef _DELHANDLERDRIFT_H_
#define _DELHANDLERDRIFT_H_

#include "startnewmodule.h"
#include "moduledrift.h"

namespace DRIFT{

  class DelHandlerDRIFT : public InteractionHandler {
  public:
    DelHandlerDRIFT(ModuleDRIFT *mod);
    ~DelHandlerDRIFT();

  protected:
    ModuleDRIFT *mod;
    void OnLeftClick(int p);
    void OnRightClick(int p);
    void OnMiddleClick(int p);
  
    void OnLeftDrag(int p, int q);
    void OnRightDrag(int p, int q);

    void OnLeftRelease(int p);

    void OnMouseMotion(int p);
    void OnEnterWindow();
    void OnLeaveWindow();
  };
} //end DRIFT namespace

#endif

