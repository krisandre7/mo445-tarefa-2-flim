
#ifndef _DRIFTDIALOG_H_
#define _DRIFTDIALOG_H_

//#include "startnewmodule.h"
#include "moduledrift.h"
#include "addhandlerdrift.h"
#include "delhandlerdrift.h"
#include "livehandlerdrift.h"

namespace DRIFT{

class DRIFTDialog : public BaseDialog{
public:
  typedef enum {NAVIGATOR=0,
		ADDMARKER,
		DELMARKER,
    LIVEMARKER} ModeType;

  DRIFTDialog(wxWindow *parent,
		    ModuleDRIFT *mod);
  ~DRIFTDialog();

  void OnChangeMode(wxCommandEvent& event);
  void OnReset(wxCommandEvent& event);
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

  DRIFTDialog::ModeType GetOperationMode();
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
  int id_run;
  int id_add;
  int id_relax;


  AddHandlerDRIFT         *xhandler;
  AddHandlerDRIFT         *yhandler;
  AddHandlerDRIFT         *zhandler;
  DelHandlerDRIFT         *dhandler;
  LiveHandlerDRIFT        *live_xhandler;
  LiveHandlerDRIFT        *live_yhandler;
  LiveHandlerDRIFT        *live_zhandler;

  ModuleDRIFT  *mod;

  BitmapRadioButton  *but;
  BrushPicker        *bPicker;
  wxButton           *run;
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
	      DRIFTDialog  *dialog);
  ~ObjBkgPanel();
  void OnMouseEvent(wxMouseEvent& event);

private:
  DRIFTDialog  *dialog;
DECLARE_EVENT_TABLE()
  };

//--------------------------------

class ObjLabel : public wxStaticText{
public:
  ObjLabel(wxWindow *parent,
	     wxString& label,
	   DRIFTDialog  *dialog);
  ~ObjLabel();
  void OnMouseEvent(wxMouseEvent& event);

private:
  DRIFTDialog  *dialog;
DECLARE_EVENT_TABLE()
  };


} //end DRIFT namespace

#endif

