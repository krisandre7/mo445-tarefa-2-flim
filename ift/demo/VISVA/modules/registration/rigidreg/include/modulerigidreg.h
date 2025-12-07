
#ifndef _MODULERIGIDREG_H_
#define _MODULERIGIDREG_H_

#include "startnewmodule.h"
#include "rigidreg_dialog.h"
#include "color_reg.h"


namespace RigidReg
{
 
  

 class ModuleRigidReg : public RegistrationModule
 {
    
  public:
    ModuleRigidReg();
    ~ModuleRigidReg();
    
    void Start();
    bool Stop();

    //PopUpView *popup_view;

    
  protected:
    wxDialog *dialog;

    
  };





} //end RigidReg namespace



#endif

