
#ifndef _INTERACTIVEOPT_H_
#define _INTERACTIVEOPT_H_

#include "startnewmodule.h"
#include "moduleinteractive.h"
#include "addhandler.h"
#include "delhandler.h"

namespace Interactive{

  class InteractiveOpt : public BasePanel {
  public:
    typedef enum {NAVIGATOR=0,
		  ADDMARKER, 
		  DELMARKER} ModeType;

    InteractiveOpt(wxWindow *parent, 
		   ModuleInteractive *mod);
    ~InteractiveOpt();
    void OnChangeMode(wxCommandEvent& event);
    void OnReset(wxCommandEvent& event);
    void OnRun(wxCommandEvent& event);
    void OnFinish(wxCommandEvent& event);
    InteractiveOpt::ModeType GetOperationMode();
    AdjRel   *GetBrush();
    wxCursor *GetBrushCursor(int zoom);
    void      NextBrush();
    void      PrevBrush();

  private:
    int id_res;
    int id_but;
    int id_bp;
    int id_run;
    int id_fin;

    AddHandler         *xhandler;
    AddHandler         *yhandler;
    AddHandler         *zhandler;
    DelHandler         *dhandler;
    ModuleInteractive  *mod;

    wxButton           *res;
    BitmapRadioButton  *but;
    BrushPicker        *bPicker;
    wxButton           *run;
    wxButton           *finish;
  };
} //end Interactive namespace

#endif

