
#ifndef _MANUALOPT_H_
#define _MANUALOPT_H_

#include "startnewmodule.h"
#include "modulemanual.h"
#include "painthandler.h"

namespace Manual{

  class ManualOpt : public BasePanel {
  public:
    typedef enum {NAVIGATOR=0,
		  PAINT} ModeType;

    ManualOpt(wxWindow *parent,
	      ModuleManual *mod);
    ~ManualOpt();
    void OnChangeMode(wxCommandEvent& event);
    void OnReset(wxCommandEvent& event);
    void OnFinish(wxCommandEvent& event);
    ManualOpt::ModeType GetOperationMode();
    AdjRel   *GetBrush();
    wxCursor *GetBrushCursor(int zoom);
    void      NextBrush();
    void      PrevBrush();

  private:
    int id_res;
    int id_but;
    int id_bp;
    int id_fin;
    PaintHandler       *xhandler;
    PaintHandler       *yhandler;
    PaintHandler       *zhandler;
    ModuleManual       *mod;

    wxButton           *res;
    BitmapRadioButton  *but;
    BrushPicker        *bPicker;
    wxButton           *finish;
  };
} //end Manual namespace

#endif


