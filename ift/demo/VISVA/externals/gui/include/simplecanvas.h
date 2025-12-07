
#ifndef _SIMPLECANVAS_H_
#define _SIMPLECANVAS_H_

#include "wxgui.h"
extern "C" {
#include "oldift.h"
}

class SimpleCanvas : public wxPanel {
public:
  int imageWidth, imageHeight;

  SimpleCanvas(wxWindow *parent);
  void OnPaint(wxPaintEvent& event);
  void DrawCImage(CImage *cimg);
  void DrawFigure(wxBitmap bitmap);
  void ClearFigure();
  void Refresh();
  
private:
  wxWindow *owner;
  wxMemoryDC *ibuf;
  wxImage *wximg;
  bool loaded;
  
  DECLARE_EVENT_TABLE()
};

#include "gui.h"

#endif
