
#ifndef _MODULESETORIENTATION_H_
#define _MODULESETORIENTATION_H_

#include "startnewmodule.h"
#include "setorientationdialog.h"

namespace Setorientation{

  class ModuleSetorientation : public PreProcModule{
  public:
    ModuleSetorientation();
    ~ModuleSetorientation();
    void Start();
    bool Stop();
    //Scene* ChangeOrientationToLPS(Scene *scn, char *ori);

  protected:
    wxDialog *dialog;

  };
} //end Setorientation namespace

#endif

