
#ifndef _SLICEVIEWOPT_H_
#define _SLICEVIEWOPT_H_

#include "startnewmodule.h"
#include "modulesliceview.h"

namespace SliceView{

  class SliceViewOpt : public BasePanel {
  public:
    SliceViewOpt(wxWindow *parent,
		 ModuleSliceView *mod);
    ~SliceViewOpt();
    void OnChangeChoice(wxCommandEvent& event);
  private:
    ModuleSliceView *mod;
    wxChoice        *chHighlight;
    wxChoice        *chData;
    wxChoice        *chMarker;
    int id_high;
    int id_data;
    int id_mark;
  };
} //end SliceView namespace

#endif

