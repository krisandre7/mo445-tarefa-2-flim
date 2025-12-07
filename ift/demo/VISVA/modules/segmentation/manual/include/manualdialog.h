
#ifndef _MANUALDIALOG_H_
#define _MANUALDIALOG_H_

#include "startnewmodule.h"
#include "modulemanual.h"
#include "painthandler.h"

namespace Manual{

class ManualDialog : public BaseDialog{
public:
  typedef enum {NAVIGATOR=0,
		PAINT} ModeType;

  ManualDialog(wxWindow *parent,
	       ModuleManual *mod);
  ~ManualDialog();

  void OnChangeMode(wxCommandEvent& event);
  void OnReset(wxCommandEvent& event);
  void OnUndo(wxCommandEvent& event);

  void OnAddObj(wxCommandEvent& event);
  void OnDelete(wxCommandEvent& event);
  void OnChangeColor(wxCommandEvent& event);
  void OnChangeVisibility(wxCommandEvent& event);

  void OnCancel(wxCommandEvent& event);
  void OnOk(wxCommandEvent& event);
  void OnSize(wxSizeEvent& event);
  void ChangeObjSelection(wxPanel *objbkg);

  ManualDialog::ModeType GetOperationMode();
  AdjRel   *GetBrush();
  wxCursor *GetBrushCursor(int zoom);
  void      NextBrush();
  void      PrevBrush();

private:
  wxScrolledWindow *CreateObjectPanel();
  void              DestroyObjectPanel();

  int id_but;
  int id_bp;
  int id_res;
  int id_undo;
  int id_add;
  PaintHandler       *xhandler;
  PaintHandler       *yhandler;
  PaintHandler       *zhandler;
  ModuleManual       *mod;

  BitmapRadioButton  *but;
  BrushPicker        *bPicker;
  wxButton           *undo;
  wxButton           *res;
  wxButton           *addobj;
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
	      ManualDialog  *dialog);
  ~ObjBkgPanel();
  void OnMouseEvent(wxMouseEvent& event);

private:
  ManualDialog  *dialog;
DECLARE_EVENT_TABLE()
  };

//--------------------------------

class ObjLabel : public wxStaticText{
public:
  ObjLabel(wxWindow *parent,
	     wxString& label,
	   ManualDialog  *dialog);
  ~ObjLabel();
  void OnMouseEvent(wxMouseEvent& event);

private:
  ManualDialog  *dialog;
DECLARE_EVENT_TABLE()
  };


} //end Manual namespace

#endif

