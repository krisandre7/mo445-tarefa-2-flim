
#ifndef _INTERPOLATIONDIALOG_H_
#define _INTERPOLATIONDIALOG_H_

#include "startnewmodule.h"

namespace Interpolation{

  class CustomDialog : public wxDialog{
  public:
    CustomDialog();
    ~CustomDialog();
    void GetVoxelSize(float *dx, float *dy, float *dz);
  protected:
    SpinFloat *sdx;
    SpinFloat *sdy;
    SpinFloat *sdz;
  };


  class IsotropicDialog : public wxDialog{
  public:
    IsotropicDialog();
    ~IsotropicDialog();
    void OnCustom(wxCommandEvent& event);
  protected:
    int id_cus;
  };

} //end Interpolation namespace

#endif

