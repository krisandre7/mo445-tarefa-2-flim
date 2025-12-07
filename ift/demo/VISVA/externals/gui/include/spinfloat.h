
#ifndef _SPINFLOAT_H_
#define _SPINFLOAT_H_

#include "gui.h"

class SpinFloat : public wxPanel{
public:
  SpinFloat(wxWindow* parent, wxWindowID id,
	    float min, float max, float initial,
	    float increment, int digits);
  ~SpinFloat();
  float GetMax();
  float GetMin();
  float GetValue();
  void  SetValue(float value);
  void  SetRange(float minValue, float maxValue);
  void  OnPress(wxCommandEvent& event);
  void  OnChange(wxCommandEvent& event);
  void  OnEnter(wxCommandEvent& event);
  void  ValidateContents();
private:
  float value;
  float minValue;
  float maxValue;
  float increment;
  int   digits;
  FloatCtrl *ctrl;
  wxBitmapButton *up;
  wxBitmapButton *down;
};

#endif
