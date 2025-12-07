
#ifndef _BRUSHPICKER_H_
#define _BRUSHPICKER_H_

#include "simplecanvas.h"
#include "gui.h"


class BrushPicker : public wxPanel {
public:
  BrushPicker(wxWindow *parent,
	      wxWindowID id, bool initdefault);
  ~BrushPicker();
  AdjRel   *GetBrush();
  void      AddBrush(AdjRel *brush);
  void      NextBrush();
  void      PrevBrush();
  void      Refresh();
  void      OnPress(wxCommandEvent & event);
  wxCursor *GetBrushCursor(int zoom);

private:

  int n;
  int selection;
  int maxWidth, maxHeight;
  wxWindow *owner;
  wxBitmapButton *up;
  wxBitmapButton *down;
  SimpleCanvas *scanvas;
  AdjRel **brushes;

};

#endif
