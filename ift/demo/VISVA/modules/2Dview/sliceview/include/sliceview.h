
#ifndef _SLICEVIEW_H_
#define _SLICEVIEW_H_

#include "startnewmodule.h"
#include "slicecanvas.h"

namespace SliceView{

  class SliceView : public wxPanel {
  public:
    SliceCanvas *canvas;
    wxSpinCtrl *spinSlice;
    int ntimes;

    SliceView(wxWindow *parent, char axis);
    ~SliceView();
    void  SliceRefresh();
    void  SetCutVoxel(Voxel Cut);
    Voxel GetCutVoxel();
    int   GetSliceNumber();
    void  ShowCoordinate(Voxel *v);
    void  OnSliceRotate(wxCommandEvent & event);
    void  OnSliceSpin(wxSpinEvent& event);
    void  OnSaveCanvasView(wxCommandEvent & event);
    void  EnableBalloon();
    void  DisableBalloon();
    void  SetBalloonText(char *text);
    void  MoveBalloon(int x, int y);

  private:
    static Voxel Cut;
    
    wxWindow       *owner;
    wxStaticText   *coordinate;
    wxBitmapButton *butRotate;  
    wxBitmapButton *butSave;
    //wxToolTip      *balloon;
    wxFrame        *balloon;
    wxStaticText   *balloontext;
  };
} //end SliceView namespace


#endif

