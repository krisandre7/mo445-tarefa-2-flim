
#ifndef _ENHANCEMENTDIALOG_H_
#define _ENHANCEMENTDIALOG_H_

#include "startnewmodule.h"
#include "moduleenhancement.h"
#include "addsamples.h"
#include "delsamples.h"

namespace Enhancement{

class EnhancementDialog : public BaseDialog{
public:
  typedef enum {NAVIGATOR=0,
		ADDMARKER, 
		DELMARKER} ModeType;

  EnhancementDialog(wxWindow *parent,
		    ModuleEnhancement *mod);
  ~EnhancementDialog();

  void OnChangeMode(wxCommandEvent& event);
  void OnReset(wxCommandEvent& event);
  void OnUndo(wxCommandEvent& event);
  void OnRun(wxCommandEvent& event);

  void OnCancel(wxCommandEvent& event);
  void OnOk(wxCommandEvent& event);
  void OnSize(wxSizeEvent& event);

  EnhancementDialog::ModeType GetOperationMode();
  AdjRel   *GetBrush();
  wxCursor *GetBrushCursor(int zoom);
  void      NextBrush();
  void      PrevBrush();

private:
  int id_but;
  int id_bp;
  int id_res;
  int id_undo;
  int id_run;

  AddSamples         *xhandler;
  AddSamples         *yhandler;
  AddSamples         *zhandler;
  DelSamples         *dhandler;
  ModuleEnhancement  *mod;

  BitmapRadioButton  *but;
  BrushPicker        *bPicker;
  wxButton           *run;
  wxButton           *undo;
  wxButton           *res;
  BasePanel  *panel;

DECLARE_EVENT_TABLE()
  };


} //end Enhancement namespace

#endif

