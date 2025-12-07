
#ifndef _BITMAPTOGGLEBUTTON_H_
#define _BITMAPTOGGLEBUTTON_H_

#include "wxgui.h"

class BitmapToggleButton : public wxBitmapButton {
public:
  BitmapToggleButton(wxWindow *parent,
		     wxWindowID id,
		     wxBitmap *bitmap0,
		     wxBitmap *bitmap1);
  ~BitmapToggleButton();
  void OnPress(wxCommandEvent& event);
  bool GetValue();
  void SetValue(bool state);
  void ToggleValue();
  void SetBkgColor(bool state, int color);
private:
  bool state;
  wxBitmap *bitmap0;
  wxBitmap *bitmap1;
  int color0;
  int color1;
};

#include "gui.h"

#endif

