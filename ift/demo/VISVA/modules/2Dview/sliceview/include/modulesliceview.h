
#ifndef _MODULESLICEVIEW_H_
#define _MODULESLICEVIEW_H_

#include "startnewmodule.h"
#include "sliceview.h"

namespace SliceView{

  class ModuleSliceView : public View2DModule{
  public:
    ModuleSliceView();
    ~ModuleSliceView();
    wxPanel *GetViewPanel(wxWindow *parent,
			  char axis);
    wxPanel *GetCustomViewPanel(wxWindow *parent,
				Scene *scn,
				char axis);
    void DelViewPanel(wxPanel **view);
    void SetInteractionHandler(InteractionHandler *handler,
			       char axis);
    void  Start();
    bool  Stop();
    void  Refresh();
    void  Refresh(char axis);
    void  SetRefreshHandler(RefreshHandler *handler);
    void  Clear();
    void  Zoomin();
    void  Zoomout();
    void  ChangeDrawMarker();
    void  SetZoomLevel(float zoom);
    float GetZoomLevel();
    void  Set2DViewOptions(HighlightType highlight,
			   DataType data,
			   MarkerType marker);
    void  SetSliceVoxel(Voxel Cut);
    Voxel GetSliceVoxel();
    CImage *CopyAsCImage(char axis);
    void SetCursor(  wxCursor *cursor, char axis);
    
  protected:
    ArrayList *Panels;
    wxPanel   *optPanel;
  };
}

#include "sliceviewopt.h"

#endif

