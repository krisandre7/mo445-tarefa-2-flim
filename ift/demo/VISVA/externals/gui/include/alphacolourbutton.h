
#ifndef _ALPHACOLOURBUTTON_H_
#define _ALPHACOLOURBUTTON_H_

#include "wxgui.h"

class AlphaColourButton : public wxPanel {
public:
  AlphaColourButton(wxWindow *parent,
		    wxWindowID id);
  ~AlphaColourButton();
  void OnPress(wxCommandEvent& event);
  void Refresh();
  int  GetValue();
  void SetValue(int color);
  int  GetAlpha();
  void SetAlpha(int alpha);

private:
  void ChangeAlpha();
  int color;
  int alpha;
  wxWindow *owner;
  wxBitmapButton *but;
  wxBitmapButton *up;
  wxBitmapButton *down;
  int *samples;
};

#include "gui.h"

#endif

