
#ifndef _MODULESELFREG_H_
#define _MODULESELFREG_H_

#include "startnewmodule.h"

#include "selfreg_dialog.h"

namespace SelfReg
{
 
  

 class ModuleSelfReg : public RegistrationModule
 {
    
  public:
    ModuleSelfReg();
    ~ModuleSelfReg();
    
    void Start();
    bool Stop();

    //PopUpView *popup_view;

    
  protected:
    wxDialog *dialog;

    
  };





} //end SelfReg namespace



#endif

