
#ifndef _BRICONTRDIALOG_H_
#define _BRICONTRDIALOG_H_

#include "gui.h"


class BriContrDialog : public wxDialog{
public:
  BriContrDialog(wxWindow *parent, 
		 wxWindowID id);
  ~BriContrDialog();
  void GetBCLevel(int *B, int *C);
  void SetBCLevel(int B, int C);
  void ShowWindow();
  void OnCancel(wxCommandEvent& event);
private:
  wxBoxSizer   *vsizer;
  wxSlider     *sContr;
  wxSlider     *sBrigh;
  int B_old,C_old;
    };

    /*
BEGIN_EVENT_TABLE(BriContrDialog, wxDialog)
  EVT_BUTTON(wxID_CANCEL,           BriContrDialog::OnCancel)
  EVT_COMMAND_SCROLL(ID_Contrast,   BriContrDialog::OnChangeBC)
  EVT_COMMAND_SCROLL(ID_Brightness, BriContrDialog::OnChangeBC)
  END_EVENT_TABLE()
    */

#endif

