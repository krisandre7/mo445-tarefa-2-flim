
#ifndef _RIGIDREG_DIALOG_H_
#define _RIGIDREG_DIALOG_H_

#include "startnewmodule.h"

namespace RigidReg {

  class RigidRegDialog : public wxDialog{
    //class RigidRegDialog : public BaseDialog{


  public:
    RigidRegDialog(Scene *R, Scene *G, Scene *B);
    ~RigidRegDialog();
    int GetOrientation();
    int GetSpinValue();
    void OnChangeSliceOrientation(wxCommandEvent& event);
    void OnChangeSpinCtrl(wxCommandEvent& event);


  protected:
    wxChoice *chOri;
    wxSpinCtrl *spinSlice;
    Canvas *canvas;

    Scene *R,*G,*B;

    //wxChoice *acuOri;
  };



} 



#endif

