
#ifndef _MODULEMSP_H_
#define _MODULEMSP_H_

#include "startnewmodule.h"
#include "msp_dialog.h"
#include "msp_align.h"

namespace MSP{

  class ModuleMSP : public PreProcModule{
  public:
    ModuleMSP();
    ~ModuleMSP();
    void Start();
    bool Stop();

  protected:
    wxDialog *dialog;
  };
} //end MSP namespace

#endif

