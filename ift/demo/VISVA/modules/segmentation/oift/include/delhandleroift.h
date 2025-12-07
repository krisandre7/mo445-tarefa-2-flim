
#ifndef _DELHANDLEROIFT_H_
#define _DELHANDLEROIFT_H_

#include "startnewmodule.h"
#include "moduleoift.h"

namespace OIFT{

  class DelHandlerOIFT : public InteractionHandler {
  public:
    DelHandlerOIFT(ModuleOIFT *mod);
    ~DelHandlerOIFT();

  protected:
    ModuleOIFT *mod;
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
} //end OIFT namespace

#endif

