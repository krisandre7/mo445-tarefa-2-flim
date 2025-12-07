
#ifndef _SPINSLIDER_H_
#define _SPINSLIDER_H_

#include "gui.h"


class SpinSlider : public wxPanel{
public:
  SpinSlider(wxWindow* parent, wxWindowID id,
	     int value , int minValue, int maxValue);
  ~SpinSlider();
  int  GetMax();
  int  GetMin();
  int  GetValue();
  void SetValue(int value);
  void SetRange(int minValue, int maxValue);
  void OnScroll(wxScrollEvent& event);
  void OnSpin(wxSpinEvent& event);
private:
  wxSlider   *slider;
  wxSpinCtrl *spin;
};

#endif

