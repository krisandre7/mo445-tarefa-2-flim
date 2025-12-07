
#ifndef _SETORIENTATIONDIALOG_H_
#define _SETORIENTATIONDIALOG_H_

#include "startnewmodule.h"

namespace Setorientation{

  class SetorientationDialog : public wxDialog{
  public:
    SetorientationDialog();
    ~SetorientationDialog();

    void GetOrientationString(char s[4]);

    int GetChoiceX();
    int GetChoiceY();
    int GetChoiceZ();
 
  protected:
    Canvas *canvasX;
    Canvas *canvasY;
    Canvas *canvasZ;
    wxSpinCtrl *spinSliceX;
    wxSpinCtrl *spinSliceY;
    wxSpinCtrl *spinSliceZ;
    wxChoice *choiceX;
    wxChoice *choiceY;
    wxChoice *choiceZ;
    void UpdateCanvas();
    void AutoLeftRight();
    void OnChangeSpinCtrl(wxCommandEvent& event);
    void OnChangeChoiceX(wxCommandEvent& event);
    void OnChangeChoiceY(wxCommandEvent& event);
    void OnChangeChoiceZ(wxCommandEvent& event);
  };


} //end Setorientation namespace

#endif

