
#ifndef _COLOURBUTTON_H_
#define _COLOURBUTTON_H_

#include "wxgui.h"

class ColourButton : public wxBitmapButton {
public:
  ColourButton(wxWindow *parent,
	       wxWindowID id);
  ~ColourButton();
  void OnPress(wxCommandEvent& event);
  int  GetValue();
  void SetValue(int color);
  
private:
  int color;
};

#include "gui.h"

#endif

