
#ifndef _VENTROPT_H_
#define _VENTROPT_H_

#include "startnewmodule.h"
#include "moduleventr.h"

namespace Ventr{

  class AddHandler : public Basic::AddHandler {
  public:
    AddHandler(ModuleVentr *mod,
	       char axis, BrushPicker *brush)
      : Basic::AddHandler(axis,brush){this->mod = mod;}
    ~AddHandler(){}
    void OnMiddleClick(int p){ mod->Run(); }
    void OnMouseMotion(int p){ APP->Window->SetStatusText(_T("Mouse Left: Add Object Marker, Mouse Center: Run, Mouse Right: Add Background Marker")); }
  private:
    ModuleVentr  *mod;
  };

  class DelHandler : public Basic::DelHandler {
  public:
    DelHandler(ModuleVentr *mod)
      : Basic::DelHandler(){this->mod = mod;}
    ~DelHandler(){}
    void OnMiddleClick(int p){ mod->Run(); }
    void OnMouseMotion(int p){ APP->Window->SetStatusText(_T("Mouse Left: Select Marker for Removal, Mouse Center: Run")); }
  private:
    ModuleVentr  *mod;
  };


  class VentrOpt : public BasePanel {
  public:
    typedef enum {NAVIGATOR=0,
		  ADDMARKER, 
		  DELMARKER} ModeType;

    VentrOpt(wxWindow *parent, 
	     ModuleVentr *mod);
    ~VentrOpt();
    void OnChangeMode(wxCommandEvent& event);
    void OnReset(wxCommandEvent& event);
    void OnFinish(wxCommandEvent& event);
    VentrOpt::ModeType GetOperationMode();
    AdjRel *GetBrush();

  private:
    int id_res;
    int id_but;
    int id_bp;
    int id_fin;
    AddHandler *xhandler;
    AddHandler *yhandler;
    AddHandler *zhandler;
    DelHandler *dhandler;
    ModuleVentr  *mod;

    wxButton           *res;
    BitmapRadioButton  *but;
    BrushPicker        *bPicker;
    wxButton           *finish;
  };
} //end Ventr namespace

#endif

