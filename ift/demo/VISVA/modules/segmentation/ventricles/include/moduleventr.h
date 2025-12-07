
#ifndef _MODULEVENTR_H_
#define _MODULEVENTR_H_

#include "startnewmodule.h"

namespace Ventr{

  class ModuleVentr : public SegmentationModule{
  public:
    ModuleVentr();
    ~ModuleVentr();
    void Start();
    bool Stop();
    //void Accept();
    void Finish();
    void Run();
    void Reset();
    AdjRel *GetBrush();

    char obj_name[1024];
    Scene *csf;

  protected:

    wxPanel *optPanel;
  };
} //end Ventr namespace

#endif

