
#ifndef _BASEPANEL_H_
#define _BASEPANEL_H_

#include "wxgui.h"

class BasePanel : public wxPanel {
public:
  BasePanel(wxWindow *parent);
  virtual ~BasePanel();

  wxSizer  *sizer;  
protected:
};

#include "gui.h"

#endif

