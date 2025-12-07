
#ifndef _MODULETHRESHOLD_H_
#define _MODULETHRESHOLD_H_

#include "startnewmodule.h"


namespace ModThreshold{

  class ModuleThreshold : public SegmentationModule{
  public:
    ModuleThreshold();
    ~ModuleThreshold();
    void Start();
    bool Stop();

    char obj_name[1024];
    int  obj_color;
  protected:
    BaseDialog *TDialog;
    int  id1,id2,id3;
  };

} //end ModThreshold namespace

#endif


