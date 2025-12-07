
#ifndef _BITMAPRADIOBUTTON_H_
#define _BITMAPRADIOBUTTON_H_

#include "wxgui.h"
#include "bitmaptogglebutton.h"

class BitmapRadioButton : public wxPanel {
public:
  BitmapRadioButton(wxWindow *parent,
		    wxWindowID id,
		    wxBitmap *bitmap[],
		    int n);
  ~BitmapRadioButton();
  void OnPress(wxCommandEvent& event);
  int  GetSelection();
  void SetSelection(int sel);
  void DisableButton(int pos);
  void EnableButton(int pos);

private:
  wxWindow *owner;
  wxWindowID id;

  wxBoxSizer *hbs;
  BitmapToggleButton **v_but;
  int n;
};

#endif

