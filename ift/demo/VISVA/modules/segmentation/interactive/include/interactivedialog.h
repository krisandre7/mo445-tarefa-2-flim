
#ifndef _INTERACTIVEDIALOG_H_
#define _INTERACTIVEDIALOG_H_

#include "startnewmodule.h"
#include "moduleinteractive.h"
#include "addhandler.h"
#include "delhandler.h"
#include "livehandler.h"

namespace Interactive{

class InteractiveDialog : public BaseDialog{
public:
  typedef enum {NAVIGATOR=0,
		ADDMARKER,
		DELMARKER,
                LIVEMARKER} ModeType;

  InteractiveDialog(wxWindow *parent,
		    ModuleInteractive *mod);
  ~InteractiveDialog();

  void OnChangeMode(wxCommandEvent& event);
  void OnReset(wxCommandEvent& event);
  void OnUndo(wxCommandEvent& event);
  void OnRun(wxCommandEvent& event);
  void OnRelax(wxCommandEvent& event);

  void OnAddObj(wxCommandEvent& event);
  void OnDelete(wxCommandEvent& event);
  void OnChangeColor(wxCommandEvent& event);
  void OnChangeVisibility(wxCommandEvent& event);

  void OnCancel(wxCommandEvent& event);
  void OnOk(wxCommandEvent& event);
  void OnSize(wxSizeEvent& event);
  void ChangeObjSelection(wxPanel *objbkg);

  InteractiveDialog::ModeType GetOperationMode();
  AdjRel   *GetBrush();
  wxCursor *GetBrushCursor(int zoom);
  void      NextBrush();
  void      PrevBrush();

private:
  wxScrolledWindow *CreateObjectPanel();
  void              DestroyObjectPanel();
  void              UnmarkAllForRemoval();

  int id_but;
  int id_bp;
  int id_res;
  int id_undo;
  int id_run;
  int id_add;
  int id_relax;

  AddHandler         *xhandler;
  AddHandler         *yhandler;
  AddHandler         *zhandler;
  DelHandler         *dhandler;
  LiveHandler        *live_xhandler;
  LiveHandler        *live_yhandler;
  LiveHandler        *live_zhandler;

  ModuleInteractive  *mod;

  BitmapRadioButton  *but;
  BrushPicker        *bPicker;
  //wxSpinCtrl         *relaxIterations;
  wxButton           *run;
  wxButton           *undo;
  wxButton           *res;
  wxButton           *addobj;
  wxButton           *relax;
  BasePanel  *panel;

  //ObjectPanel:
  wxScrolledWindow *objPanel;
  wxPanel            **v_panel_bkg;
  AlphaColourButton  **v_but_color;
  BitmapToggleButton **v_but_eye;
  wxBitmapButton     **v_but_trash;

DECLARE_EVENT_TABLE()
  };

//--------------------------------

class ObjBkgPanel : public wxPanel{
public:
  ObjBkgPanel(wxWindow *parent,
	        wxPoint& pos,
	        wxSize& size,
	      InteractiveDialog  *dialog);
  ~ObjBkgPanel();
  void OnMouseEvent(wxMouseEvent& event);

private:
  InteractiveDialog  *dialog;
DECLARE_EVENT_TABLE()
  };

//--------------------------------

class ObjLabel : public wxStaticText{
public:
  ObjLabel(wxWindow *parent,
	     wxString& label,
	   InteractiveDialog  *dialog);
  ~ObjLabel();
  void OnMouseEvent(wxMouseEvent& event);

private:
  InteractiveDialog  *dialog;
DECLARE_EVENT_TABLE()
  };


} //end Interactive namespace

#endif

