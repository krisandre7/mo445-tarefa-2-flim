
#ifndef _MODULEPANEL_H_
#define _MODULEPANEL_H_

#include "module.h"

class ModulePanel : public BasePanel {
public:
  ModulePanel(wxWindow *parent,
	      wxWindowID id);
  ~ModulePanel();
  void    AddModule(Module *m);
  Module *GetSelection();
private:
  wxRadioBox *radio;
  wxArrayString *nameArray;
  ArrayList     *moduleArray;
};

#endif

