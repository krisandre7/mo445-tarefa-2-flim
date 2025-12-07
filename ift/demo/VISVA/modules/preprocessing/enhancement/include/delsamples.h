
#ifndef _DELSAMPLES_H_
#define _DELSAMPLES_H_

#include "startnewmodule.h"
#include "moduleenhancement.h"

namespace Enhancement{

  class DelSamples : public InteractionHandler {
  public:
    DelSamples(ModuleEnhancement *mod);
    ~DelSamples();

  protected:
    ModuleEnhancement *mod;
    void OnLeftClick(int p);
    void OnRightClick(int p);
    void OnMiddleClick(int p);
  
    void OnLeftDrag(int p, int q);
    void OnRightDrag(int p, int q);
  
    void OnMouseMotion(int p);
    void OnEnterWindow();
    void OnLeaveWindow();
  };
} //end Enhancement namespace

#endif

