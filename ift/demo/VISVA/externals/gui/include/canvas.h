
#ifndef _CANVAS_H_
#define _CANVAS_H_

#include "wxgui.h"
extern "C" {
#include "oldift.h"
}

#include "interactionhandler.h"

class Canvas : public wxScrolledWindow {
public:
  Canvas(wxWindow *parent);
  ~Canvas();
  void OnPaint(wxPaintEvent& event);
  virtual void OnMouseEvent(wxMouseEvent& event);
  void DrawCImage(CImage *cimg);
  void ClearFigure();
  virtual void Draw();
  //Returns pixel/voxel address.
  virtual int  Canvas2Address(int x, int y); 
  void  Refresh();
  void  Zoomin();
  void  Zoomout();
  void  SetZoomLevel(float zoom);
  float GetZoomLevel();
  void  SetInteractionHandler(InteractionHandler *handler);
  void  GetImageSize(int* width, int* height);
  void  GetDisplaySize(int* width, int* height);
  CImage *CopyAsCImage();
  wxImage *GetImage();

  wxMemoryDC *ibuf;

protected:
  float zoom;
  int imageWidth, imageHeight;
  int displayWidth, displayHeight;
  InteractionHandler *handler;
  wxWindow *owner;
  wxImage *wximg;

private:
  DECLARE_EVENT_TABLE()
}; 

#include "gui.h"

#endif

