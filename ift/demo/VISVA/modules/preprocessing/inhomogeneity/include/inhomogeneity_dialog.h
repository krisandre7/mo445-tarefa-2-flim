
#ifndef _INHOMOGENEITY_DIALOG_H_
#define _INHOMOGENEITY_DIALOG_H_

#include "startnewmodule.h"

namespace Inhomogeneity {

  class InhomogeneityDialog : public wxDialog{

  public:
    InhomogeneityDialog();
    ~InhomogeneityDialog();
    //int GetProtocol();
    //int GetCompression();
    //int GetKeep();
    Scene* GetObject();
    float GetFunc();
    float GetRadius();
    void OnAdvOptions(wxCommandEvent& event);

  protected:
    wxChoice *chObj;
    //wxChoice *chProto;
    //wxChoice *chComp;
    wxBoxSizer *mainsizer;
    wxGridSizer *sizer;
    wxGridSizer *sizer2;
    wxStaticText *tmsg0;
    wxStaticText *tmsg3;
    wxStaticText *tmsg4;
    wxCheckBox *adv_options;
    SpinFloat *func;
    SpinFloat *radius;
    wxStaticLine *sline;
    wxSizer *sbutton;
    //wxCheckBox *keep;

  };



} //end namespace


#endif

