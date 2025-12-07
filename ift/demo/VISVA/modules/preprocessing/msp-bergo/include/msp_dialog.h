
#ifndef _MSP_DIALOG_H_
#define _MSP_DIALOG_H_

#include "startnewmodule.h"
#include "msp_volume.h"

namespace MSP{

  class MSPDialog : public wxDialog{
  public:
    MSPDialog();
    ~MSPDialog();
    VolumeOrientation GetOrientation();
  protected:
    wxChoice *chOri;
  };


} //end MSP namespace

#endif

