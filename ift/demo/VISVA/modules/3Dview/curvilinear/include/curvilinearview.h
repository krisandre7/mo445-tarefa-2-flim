
#ifndef _CURVILINEARVIEW_H_
#define _CURVILINEARVIEW_H_

#include "startnewmodule.h"
#include "curvilinearcanvas.h"

namespace Curvilinear{
  
  class CurvilinearView : public BasePanel {
  public:
    Canvas   *canvas;
    wxSlider *sDist;
    BitmapToggleButton *but3D;

    CurvilinearView(wxWindow *parent);
    ~CurvilinearView();
    void CurvilinearRefresh(bool dataChanged, float quality);
    void OnChangeDistance(wxScrollEvent& event);
    void OnChange3D(wxCommandEvent& event);
    void OnSaveCanvasView(wxCommandEvent & event);
    int  GetDistance();
  private:
    wxBitmapButton *butSave;
  };
} //end Curvilinear namespace

#endif

