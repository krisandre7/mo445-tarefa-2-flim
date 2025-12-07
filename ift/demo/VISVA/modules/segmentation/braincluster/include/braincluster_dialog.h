
#ifndef _BRAINCLUSTER_DIALOG_H_
#define _BRAINCLUSTER_DIALOG_H_

#include "startnewmodule.h"
#include "braincluster_spectral.h"

namespace BrainCluster{

  class BrainClusterDialog : public wxDialog{

  public:
    BrainClusterDialog();
    ~BrainClusterDialog();
    float GetSamples();
    float GetSamplesCSF();
    float GetMean_Prop();
    float GetMean_PropCSF();
    float GetAuto_Prop();
    float GetKMin();
    float GetKMax();
    float GetAuto_PropCSF();
    float GetKMinCSF();
    float GetKMaxCSF();
    Scene* GetObject();
    int GetColorCSF();
    int GetColorWM();
    int GetColorGM();
    int GetCsfCheck();
    void OnChangeObject( wxCommandEvent& event );
    void OnChangeCheckBox( wxCommandEvent& event );
    void OnChangeCheckBoxCSF( wxCommandEvent& event );
    void OnCsfCheck( wxCommandEvent& event );
    void OnAdvOptions( wxCommandEvent& event );

  protected:
    wxStaticText *txtChObj;
    wxStaticText *txtSamples;
    wxStaticText *txtSamplesCSF;
    wxStaticText *txtCSF;
    wxStaticText *txtWM;
    wxStaticText *txtGM;
    wxStaticText *txtMean_Prop;
    wxStaticText *txtKMin;
    wxStaticText *txtKMax;
    wxStaticText *txtMean_PropCSF;
    wxStaticText *txtKMinCSF;
    wxStaticText *txtKMaxCSF;
    wxStaticLine *sline;
    wxChoice *chObj, *protocol;
    SpinFloat *samples, *samplesCSF;
    SpinFloat *mean_prop,*Kmin,*Kmax;
    SpinFloat *mean_propcsf,*Kmincsf,*Kmaxcsf;
    wxCheckBox *auto_prop;
    wxCheckBox *auto_propcsf, *csfCheck;
    wxCheckBox *adv_options;
    ColourButton *butColorCSF;
    ColourButton *butColorWM;
    ColourButton *butColorGM;
    wxBoxSizer *mainsizer;
    wxGridSizer *sizer;
    wxGridSizer *sizer2;
    wxGridSizer *sizer3;
    wxBoxSizer *sizer4;
    wxGridSizer *sizer5;
    wxSizer *sbutton;
    void SetCheckBox();
    void SetCheckBoxCSF();
    void SetCsf();
    void RefreshDialog();
  };



} //end namespace


#endif

