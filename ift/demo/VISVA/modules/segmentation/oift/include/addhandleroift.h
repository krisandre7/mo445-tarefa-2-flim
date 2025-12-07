
#ifndef _ADDHANDLEROIFT_H_
#define _ADDHANDLEROIFT_H_

#include "startnewmodule.h"
#include "moduleoift.h"

namespace OIFT{

  class AddHandlerOIFT : public InteractionHandler {
  public:
    AddHandlerOIFT(char axis,
	       ModuleOIFT *mod);
    ~AddHandlerOIFT();
  
  protected:
    char axis;
    ModuleOIFT *mod;
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
} //end OIFT namespace

#endif

