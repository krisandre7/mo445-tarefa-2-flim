
#ifndef _MODULEINHOMOGENEITY_H_
#define _MODULEINHOMOGENEITY_H_

#include "startnewmodule.h"
#include "inhomogeneity_dialog.h"
#include "processors.h"

namespace Inhomogeneity
{

  //  class ModuleMSP : public RegistrationModule
  class ModuleInhomogeneity : public PreProcModule
  {
    
  public:
    ModuleInhomogeneity();
    ~ModuleInhomogeneity();
    
    void Start();
    bool Stop();
    
  protected:
    wxDialog *dialog;
    
  };



} //end namespace



#endif

