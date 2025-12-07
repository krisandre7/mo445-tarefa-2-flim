
#ifndef _ADDBASICHANDLER_H_
#define _ADDBASICHANDLER_H_

#include "startnewmodule.h"

namespace Basic{

  class AddHandler : public InteractionHandler {
  public:
    AddHandler(char axis, BrushPicker *brush);
    virtual ~AddHandler();
    
  protected:
    char axis;
    int  markerID;
    BrushPicker *brush;
    virtual void OnLeftClick(int p);
    virtual void OnRightClick(int p);
    virtual void OnMiddleClick(int p);
    
    virtual void OnLeftDrag(int p, int q);
    virtual void OnRightDrag(int p, int q);
    
    virtual void OnMouseMotion(int p);
    
    //--- more func ---
    
    int unique(int id);
  };
}

#endif

