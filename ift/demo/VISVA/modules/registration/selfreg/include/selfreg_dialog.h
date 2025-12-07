
#ifndef _SELFREG_DIALOG_H_
#define _SELFREG_DIALOG_H_

#include "startnewmodule.h"

namespace SelfReg {

  class SelfRegDialog : public wxDialog{
    //class SelfRegDialog : public BaseDialog{


  public:
    SelfRegDialog(Scene *R, Scene *G, Scene *B);
    ~SelfRegDialog();
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

