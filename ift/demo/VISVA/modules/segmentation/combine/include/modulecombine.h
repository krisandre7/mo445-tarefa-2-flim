
#ifndef _MODULECOMBINE_H_
#define _MODULECOMBINE_H_

#include "startnewmodule.h"

#define MAX_COBJS 10

namespace Combine{

  class ModuleCombine : public SegmentationModule{
  public:
    ModuleCombine();
    ~ModuleCombine();
    void Start();
    bool Stop();
    void Finish();
    void Reset();
    AdjRel   *GetBrush();
    wxCursor *GetBrushCursor(int zoom);
    void      NextBrush();
    void      PrevBrush();
    void      DeleteObj(int obj);

    int  obj_sel;
    int  bkg_sel;
    int  nobjs;
    SegmObject *myobj;

  protected:
    void AllocData();
    void FreeData();

    bool active;
    //wxPanel *optPanel;
    BaseDialog *optDialog;
  };

} //end Combine namespace

#endif


