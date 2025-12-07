
#ifndef _THRESHOLDDIALOG_H_
#define _THRESHOLDDIALOG_H_

#include "startnewmodule.h"
#include "modulethreshold.h"
extern "C" {
#include "orderlist.h"
}

namespace ModThreshold{

class ThresholdDialog : public BaseDialog{
public:
  ThresholdDialog(wxWindow *parent, 
		  wxWindowID id1,
		  wxWindowID id2,
		  wxWindowID id3,
		  int Min, int Max,
		  ModuleThreshold *mod);
  ~ThresholdDialog();
  void GetThreshold(int *lower, int *higher);
  void OnThresholdChange(wxScrollEvent& event);
  void OnIntervalChange(wxScrollEvent& event);
  void OnCancel(wxCommandEvent& event);
  void OnOk(wxCommandEvent& event);
  void OnSize(wxSizeEvent& event);
  void OnChangeHistogram(wxCommandEvent& event);

private:
  void DrawHistogram();
  void RefreshObject();

  RefreshHandler *handler;
  SimpleCanvas *plot;
  wxChoice     *histogramType;
  wxStaticText *histogramTypeTxt;
  SpinSlider   *spinLower1;
  SpinSlider   *spinHigher1;
  SpinSlider   *spinLower2;
  SpinSlider   *spinHigher2;
  ModuleThreshold *mod;
  Curve *hist;
  Curve *mhist;
  int otsu;

DECLARE_EVENT_TABLE()
  };


class ThrRefreshHandler : public RefreshHandler {
  public:
    ThrRefreshHandler(ThresholdDialog *dialog);
    ~ThrRefreshHandler();
    void OnRefresh2D(char axis);
  protected:
    ThresholdDialog *dialog;
};



} //end ModThreshold namespace

#endif

