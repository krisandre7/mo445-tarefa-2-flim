
#include "msp_dialog.h"

namespace MSP{

  MSPDialog::MSPDialog()
    : wxDialog(APP->Window, wxID_ANY, _T("Volume orientation"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("MSPDialog")){

    wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

    wxStaticText *tmsg = new wxStaticText(this, wxID_ANY, _T("Choose the current volume orientation:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxString oriChoices[3];
    oriChoices[0] = _T("Sagittal");
    oriChoices[1] = _T("Coronal");
    oriChoices[2] = _T("Axial");
    chOri = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 3, oriChoices, 0, wxDefaultValidator, _T("choice0"));
    chOri->SetSelection(0);

    wxStaticLine *sline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));
    wxSizer *sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

    sizer->Add(tmsg,    0, wxEXPAND);
    sizer->Add(chOri,   0, wxEXPAND);
    sizer->Add(sline,   0, wxEXPAND);
    sizer->Add(sbutton, 0, wxEXPAND);

    this->SetSizer(sizer, true);
    sizer->SetSizeHints(this);
    sizer->Layout();
  }

  MSPDialog::~MSPDialog(){
    delete chOri;
  }

  VolumeOrientation MSPDialog::GetOrientation(){
    int sel = chOri->GetSelection();
    if(sel<0 || sel>2)
      return UnknownOrientation;
    else
      return (VolumeOrientation)sel;
  }

} //end MSP namespace




