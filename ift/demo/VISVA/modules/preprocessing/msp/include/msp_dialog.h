
#ifndef _MSP_DIALOG_H_
#define _MSP_DIALOG_H_

#include "startnewmodule.h"

namespace MSP {

  class MSPDialog : public wxDialog{

  public:
    MSPDialog();
    ~MSPDialog();
    int GetOrientation();
    int GetAccuracy();
    Scene* GetObject();

  protected:
    wxChoice *chOri;
    wxChoice *chObj;
    wxChoice *acuOri;
  };



} //end MSP namespace


#endif

