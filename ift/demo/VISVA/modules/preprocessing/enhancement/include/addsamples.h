
#ifndef _ADDSAMPLES_H_
#define _ADDSAMPLES_H_

#include "startnewmodule.h"
#include "moduleenhancement.h"

namespace Enhancement{

  class AddSamples : public InteractionHandler {
  public:
    AddSamples(char axis,
	       ModuleEnhancement *mod);
    ~AddSamples();
  
  protected:
    char axis;
    ModuleEnhancement *mod;
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
} //end Enhancement namespace

#endif

