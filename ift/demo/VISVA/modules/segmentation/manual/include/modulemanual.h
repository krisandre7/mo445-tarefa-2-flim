
#ifndef _MODULEMANUAL_H_
#define _MODULEMANUAL_H_

#include "startnewmodule.h"

#define MAX_OBJS 100000

namespace Manual{

  class ModuleManual : public SegmentationModule{
  public:
    ModuleManual();
    ~ModuleManual();
    void Start();
    bool Stop();
    void Finish();
    void Reset();
    AdjRel   *GetBrush();
    wxCursor *GetBrushCursor(int zoom);
    void      NextBrush();
    void      PrevBrush();
    void      DeleteObj(int obj);

    char obj_name[MAX_OBJS][128];
    int  obj_color[MAX_OBJS];
    int  obj_visibility[MAX_OBJS];
    int  obj_sel;
    int  nobjs;
    //SegmObject *myobj;

  protected:
    void AllocData();
    void FreeData();

    bool active;
    //wxPanel *optPanel;
    BaseDialog *optDialog;
  };

} //end Manual namespace

#endif


