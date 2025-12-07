
#ifndef _SURFACEVIEW_H_
#define _SURFACEVIEW_H_

#include "startnewmodule.h"
#include "surfacecanvas.h"

namespace Surface{

  class SurfaceView : public BasePanel {
    public:
    Canvas *canvas;
    BitmapToggleButton *but3D;
    BitmapToggleButton *butPlanes;

    SurfaceView(wxWindow *parent);
    ~SurfaceView();
    void OnChange3D(wxCommandEvent& event);
    void OnChangeViewPlanes(wxCommandEvent& event);
    void OnSaveCanvasView(wxCommandEvent & event);
    private:
      wxBitmapButton *butSave;
  };
} //end Surface namespace

#endif

