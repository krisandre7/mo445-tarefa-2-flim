
#ifndef _PROFILEDIALOG_H_
#define _PROFILEDIALOG_H_

#include "startnewmodule.h"
#include "moduleintprofile.h"

namespace IntensityProfile{

class ProfileDialog : public BaseDialog{
public:
  ProfileDialog(wxWindow *parent,
		int *line, int n,
		ModuleIntensityProfile *mod);
  ~ProfileDialog();
  void OnCancel(wxCommandEvent& event);
  void OnOk(wxCommandEvent& event);
  void OnSize(wxSizeEvent& event);
  void SetPosition(int position);
  int  GetPosition();
  int  GetProfileLength();
  void GetPlotSize(int *w, int *h);

private:
  void DrawProfile();
  int   *voxels;
  Curve *curve1;
  Curve *curve2;
  int  position;
  Canvas *plot;
  InteractionHandler *handler;
  ModuleIntensityProfile *mod;

DECLARE_EVENT_TABLE()
  };


} //end IntensityProfile namespace

#endif

