
#ifndef _COMBINEDIALOG_H_
#define _COMBINEDIALOG_H_

#include "startnewmodule.h"
#include "modulecombine.h"
#include "cpainthandler.h"

namespace Combine{

class CombineDialog : public BaseDialog{
  public:
  typedef enum { NAVIGATOR = 0,PAINT, AND, OR, NEG, XOR } ModeType;
    
  CombineDialog(wxWindow *parent,
		ModuleCombine *mod);
  ~CombineDialog();
  
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
  void ChangeBkgSelection(wxPanel *objbkg);
  void RefreshLabel(); // Refresh objects displayed in canvas.

  CombineDialog::ModeType GetOperationMode();
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
  CpaintHandler       *xhandler;
  CpaintHandler       *yhandler;
  CpaintHandler       *zhandler;
  ModuleCombine       *mod;

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
	      CombineDialog  *dialog);
  ~ObjBkgPanel();
  void OnMouseEvent(wxMouseEvent& event);

private:
  CombineDialog  *dialog;
DECLARE_EVENT_TABLE()
  };

//--------------------------------

class ObjLabel : public wxStaticText{
public:
  ObjLabel(wxWindow *parent,
	     wxString& label,
	   CombineDialog  *dialog);
  ~ObjLabel();
  void OnMouseEvent(wxMouseEvent& event);

private:
  CombineDialog  *dialog;
DECLARE_EVENT_TABLE()
  };


} //end Combine namespace

#endif

