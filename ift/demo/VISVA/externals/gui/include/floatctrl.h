
#ifndef _FLOATCTRL_H_
#define _FLOATCTRL_H_

#include "wxgui.h"

class FloatCtrl : public wxTextCtrl{
public:
  FloatCtrl(wxWindow* parent, wxWindowID id, 
	    long style);
  ~FloatCtrl();
  void OnChar(wxKeyEvent& event);
private:
};

#include "gui.h"

#endif

