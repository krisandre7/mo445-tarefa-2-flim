
#include "interpolationdialog.h"

namespace Interpolation{

  CustomDialog::CustomDialog()
    : wxDialog(APP->Window, wxID_ANY, _T("Change voxel size?"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("CustomDialog")){

    wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

    wxStaticText *tmsg = new wxStaticText(this, wxID_ANY, _T("Choose voxel size:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxStaticText *tdx = new wxStaticText(this, wxID_ANY, _T("dx (mm)"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("dx"));
    wxStaticText *tdy = new wxStaticText(this, wxID_ANY, _T("dy (mm)"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("dy"));
    wxStaticText *tdz = new wxStaticText(this, wxID_ANY, _T("dz (mm)"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("dz"));

    sdx = new SpinFloat(this, wxID_ANY, 0.01, 5.0, 
			(APP->Data.orig)->dx, 0.01, 3);
    sdy = new SpinFloat(this, wxID_ANY, 0.01, 5.0, 
			(APP->Data.orig)->dy, 0.01, 3);
    sdz = new SpinFloat(this, wxID_ANY, 0.01, 5.0, 
			(APP->Data.orig)->dz, 0.01, 3);

    wxBoxSizer *hbsdx = new wxBoxSizer(wxHORIZONTAL);
    wxBoxSizer *hbsdy = new wxBoxSizer(wxHORIZONTAL);
    wxBoxSizer *hbsdz = new wxBoxSizer(wxHORIZONTAL);

    hbsdx->Add(tdx, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsdx->Add(sdx, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsdy->Add(tdy, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsdy->Add(sdy, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsdz->Add(tdz, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsdz->Add(sdz, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);

    wxStaticLine *sline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));
    wxSizer *sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

    sizer->Add(tmsg,    0, wxEXPAND);
    sizer->Add(hbsdx,   0, wxEXPAND);
    sizer->Add(hbsdy,   0, wxEXPAND);
    sizer->Add(hbsdz,   0, wxEXPAND);
    sizer->Add(sline,   0, wxEXPAND);
    sizer->Add(sbutton, 0, wxEXPAND);

    this->SetSizer(sizer, true);
    sizer->SetSizeHints(this);
    sizer->Layout();
  }

  CustomDialog::~CustomDialog(){
    delete sdx;
    delete sdy;
    delete sdz;
  }

  void CustomDialog::GetVoxelSize(float *dx,
				  float *dy,
				  float *dz){
    *dx = sdx->GetValue();
    *dy = sdy->GetValue();
    *dz = sdz->GetValue();
  }


  IsotropicDialog::IsotropicDialog()
    : wxDialog(APP->Window, wxID_ANY, _T("Change voxel size?"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("dialogBox")){
    char msg[1024];
    char text[512];
    Scene *scn = APP->Data.orig;

    id_cus = APP->idManager->AllocID();
    sprintf(msg,"Volume \"%s\" has voxel size: \n",APP->Data.volumename);
    sprintf(text,"%.3f x %.3f x %.3f = %.3f mm^3\n\n",scn->dx,scn->dy,scn->dz,scn->dx*scn->dy*scn->dz);
    strcat(msg, text);
    sprintf(text,"Interpolate it to isotropic voxel size?");
    strcat(msg, text);

    wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

    wxString wxmsg(msg, wxConvUTF8);
    wxStaticText *tmsg = new wxStaticText(this, -1, wxmsg, wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxButton *cus = new wxButton(this, id_cus, _T("Custom..."), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("butCustom"));
    
    wxStaticLine *sline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));
    wxSizer *sbutton = this->CreateButtonSizer(wxYES|wxNO);

#if wxMINOR_VERSION>=8
    this->SetEscapeId(wxID_NO);
#endif

    sizer->Add(tmsg,    0, wxEXPAND);
    sizer->Add(cus,     0, wxALIGN_RIGHT);
    sizer->Add(sline,   0, wxEXPAND);
    sizer->Add(sbutton, 0, wxEXPAND);

    this->SetSizer(sizer, true);
    sizer->SetSizeHints(this);
    sizer->Layout();

    Connect(id_cus, wxEVT_COMMAND_BUTTON_CLICKED,
	    wxCommandEventHandler(IsotropicDialog::OnCustom),
	    NULL, NULL);
  }


  IsotropicDialog::~IsotropicDialog(){
    Disconnect(id_cus, wxEVT_COMMAND_BUTTON_CLICKED,
	       wxCommandEventHandler(IsotropicDialog::OnCustom),
	       NULL, NULL);
    APP->idManager->FreeID(id_cus);
  }


  void IsotropicDialog::OnCustom(wxCommandEvent& event){
    float dx,dy,dz;
    CustomDialog custom;
    Scene *scn = APP->Data.orig;

    if(custom.ShowModal()==wxID_OK){
      this->EndModal(wxID_NO);

      wxBusyInfo busy(_T("Please wait, working..."));
      wxBusyCursor wait;
      APP->Window->SetStatusText(_T("Please wait - Computation in progress..."));
      wxTheApp->Yield();

      custom.GetVoxelSize(&dx, &dy, &dz);
      scn = BIA_LinearInterp(scn,dx,dy,dz,APP->segmobjs);
      APP->SetDataVolume_NoDestroy(scn);
      APP->ReallocSeedStructures();
      APP->DrawSegmObjects();
      
      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
    }
  }

} //end Interpolation namespace




