
#ifndef _CHARTWINDOW_H_
#define _CHARTWINDOW_H_

#include "gui.h"


class ChartWindow : public wxFrame {
public:
  ChartWindow(wxString title, wxPoint xy, wxSize size, wxBitmap bmp);
  void OnPaint(wxPaintEvent& event);
private:
  wxPoint xy;
  wxBitmap bmp;
  DECLARE_EVENT_TABLE();
};

#endif

