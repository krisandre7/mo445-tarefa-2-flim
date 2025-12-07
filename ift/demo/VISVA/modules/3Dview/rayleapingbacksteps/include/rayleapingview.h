
#ifndef _RAYLEAPINGVIEW_H_
#define _RAYLEAPINGVIEW_H_

#include "startnewmodule.h"
#include "rayleapingcanvas.h"

namespace Rayleaping{

  class RayleapingView : public BasePanel {
    public:
    Canvas *canvas;
    BitmapToggleButton *but3D;

    RayleapingView(wxWindow *parent);
    ~RayleapingView();
    void OnChange3D(wxCommandEvent& event);
    void OnSaveCanvasView(wxCommandEvent & event);
    private:
      wxBitmapButton *butSave;
  };
} //end Rayleaping namespace

#endif

