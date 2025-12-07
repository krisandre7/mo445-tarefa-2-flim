
#ifndef _MODULEINTERPOLATION_H_
#define _MODULEINTERPOLATION_H_

#include "startnewmodule.h"
#include "interpolationdialog.h"

namespace Interpolation{

  class ModuleInterpolation : public PreProcModule{
  public:
    ModuleInterpolation();
    ~ModuleInterpolation();
    void Start();
    bool Stop();

  protected:
    wxDialog *dialog;
  };
} //end Interpolation namespace

#endif

