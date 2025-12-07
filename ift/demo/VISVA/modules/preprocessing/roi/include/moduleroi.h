
#ifndef _MODULEROI_H_
#define _MODULEROI_H_

#include "startnewmodule.h"


namespace ModROI{

  class ModuleROI : public PreProcModule {
  public:
    ModuleROI();
    ~ModuleROI();
    void Start();
    bool Stop();

  protected:
    BaseDialog *TDialog;
    int idx, idy, idz;
  };

} //end ModROI namespace

#endif


