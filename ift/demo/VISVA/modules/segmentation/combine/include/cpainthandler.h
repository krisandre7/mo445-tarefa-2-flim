
#ifndef _CPAINTHANDLER_H_
#define _CPAINTHANDLER_H_

#include "startnewmodule.h"
#include "modulecombine.h"

namespace Combine{

  typedef enum { LEFT, RIGHT } cbutton;

  class CpaintHandler : public InteractionHandler {
  public:
    int mode; // 0 - PAINT, 1 - AND, 2 - OR, 3 - NEG, 4 - XOR
    CpaintHandler(char axis, ModuleCombine *mod);
    ~CpaintHandler();
  
  protected:
    char axis;
    ModuleCombine *mod;
    void OnLeftClick(int p);
    void OnRightClick(int p);
    void OnMiddleClick(int p);
    void OnMouseWheel(int p, int rotation, int delta);
    void OnLeftDrag(int p, int q);
    void OnRightDrag(int p, int q);
  
    void OnMouseMotion(int p);
    void OnEnterWindow();
    void OnLeaveWindow();
    // Set pixels from brush with multiple functions.
    void CombineBrushCustom(int p, int label1, int label2, int mode, AdjRel *A, char axis, cbutton bt);
  // Set pixels in trace with multiple functions.
    void CombineBrushTraceCustom(int p, int q, int label1, int label2, int mode, AdjRel *A, char axis, cbutton bt);
  };
} //end Combine namespace

#endif

