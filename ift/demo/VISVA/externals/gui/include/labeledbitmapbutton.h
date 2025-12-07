
#ifndef _LABELEDBITMAPBUTTON_H_
#define _LABELEDBITMAPBUTTON_H_

#include "wxgui.h"

class LabeledBitmapButton : public wxBitmapButton {
public:
  LabeledBitmapButton(wxWindow *parent,
		      wxWindowID id,
		        wxBitmap& bitmap,
		        wxString& label);
  ~LabeledBitmapButton();
  //void OnPress(wxCommandEvent& event);
  void SetBitmapDisabled(  wxBitmap& bitmap);
  void SetBitmapFocus(  wxBitmap& bitmap);
  void SetBitmapHover(  wxBitmap& bitmap);
  void SetBitmapLabel(  wxBitmap& bitmap);
  void SetBitmapSelected(  wxBitmap& bitmap);
  void SetLabel(  wxString& label);
  
private:
  wxString *wxstr;

  wxBitmap *LabelBitmap(  wxBitmap& bitmap,
			wxString& label);
};

#include "gui.h"

#endif

