
#ifndef _BASEDIALOG_H_
#define _BASEDIALOG_H_

#include "wxgui.h"

class BaseDialog : public wxDialog {
public:
  BaseDialog(wxWindow *parent,
	     char     *title);
  virtual ~BaseDialog();
  void AddPanel(wxPanel *panel);
  void AddPanel(wxPanel *panel, int proportion);
  void DetachPanel(wxPanel *panel);
  void ShowWindow();
  int  ShowModal();
  virtual void OnCancel(wxCommandEvent& event);
  virtual void OnOk(wxCommandEvent& event);

  wxSizer  *sizer;
  
protected:
  wxWindow *parent;
  char title[512];
};

#include "gui.h"

#endif



