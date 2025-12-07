
#include "chartwindow.h"


BEGIN_EVENT_TABLE(ChartWindow, wxFrame)
  EVT_PAINT(ChartWindow::OnPaint)
END_EVENT_TABLE()


ChartWindow :: ChartWindow(wxString title, wxPoint xyP, wxSize size, 
                             wxBitmap bmp)
    : wxFrame((wxWindow *)NULL, -1, title, xyP, size)
{
  this->xy = xyP;
  this->bmp = bmp;
}

void ChartWindow :: OnPaint(wxPaintEvent& event) {
  wxPaintDC dc(this);  
  dc.DrawBitmap(bmp, xy.x, xy.y, false);
}


