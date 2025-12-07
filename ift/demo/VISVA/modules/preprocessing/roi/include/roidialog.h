
#ifndef _ROIDIALOG_H_
#define _ROIDIALOG_H_

#include "startnewmodule.h"
#include "moduleroi.h"

namespace ModROI{


class ROIDialog : public BaseDialog{
public:
  ROIDialog(wxWindow *parent, wxWindowID idx, wxWindowID idy, wxWindowID idz, Scene *scn, ModuleROI *mod);
  ~ROIDialog();
  void OnROIChange(wxScrollEvent& event);
  void GetROI(Voxel *v0, Voxel *v1);

private:
  RefreshHandler *handler;
  ModuleROI *mod;
  wxSpinCtrl *x1;
  wxSpinCtrl *x2;
  wxSpinCtrl *y1;
  wxSpinCtrl *y2;
  wxSpinCtrl *z1;
  wxSpinCtrl *z2;
  void OnCancel(wxCommandEvent& event);
  void OnOk(wxCommandEvent& event);
  DECLARE_EVENT_TABLE()
  };


class ROIRefreshHandler : public RefreshHandler {
  public:
    ROIRefreshHandler(ROIDialog *dialog);
    ~ROIRefreshHandler();
    void OnRefresh2D(char axis);
  protected:
    ROIDialog *dialog;
  };
  


} //end ModROI namespace

#endif

