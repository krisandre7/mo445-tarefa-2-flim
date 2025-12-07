

#include "msp_dialog.h"

namespace MSP{

  MSPDialog::MSPDialog()
    : wxDialog(APP->Window, wxID_ANY, _T("Settings"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("MSPDialog")){

    wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

    wxStaticText *tmsg = new wxStaticText(this, wxID_ANY, _T("Choose the current volume orientation:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxString oriChoices[3];
    oriChoices[0] = _T("Sagittal");
    oriChoices[1] = _T("Axial");
    oriChoices[2] = _T("Coronal");
    chOri = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 3, oriChoices, 0, wxDefaultValidator, _T("choice0"));
    chOri->SetSelection(0);


    wxStaticText *tmsg2 = new wxStaticText(this, wxID_ANY, _T("Choose the accuracy/speed:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxString acuChoices[3];
    acuChoices[0] = _T("Best Accuracy (Slowest)");
    acuChoices[1] = _T("Good Accuracy (Moderate)");
    acuChoices[2] = _T("Fair Accuracy (Fastest)");
    acuOri = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 3, acuChoices, 0, wxDefaultValidator, _T("choice0"));
    acuOri->SetSelection(0);


    wxStaticLine *sline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));
    wxSizer *sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

    sizer->Add(tmsg,    0, wxEXPAND);
    sizer->Add(chOri,   0, wxEXPAND);
    sizer->Add(tmsg2,    0, wxEXPAND);
    sizer->Add(acuOri,   0, wxEXPAND);
    sizer->Add(sline,   0, wxEXPAND);
    sizer->Add(sbutton, 0, wxEXPAND);

    this->SetSizer(sizer, true);
    sizer->SetSizeHints(this);
    sizer->Layout();
  }

  MSPDialog::~MSPDialog(){
    delete chOri;
    delete acuOri;
  }

  
  int MSPDialog::GetOrientation(){
    return chOri->GetSelection()+1;
  }

  int MSPDialog::GetAccuracy(){
    return acuOri->GetSelection()+1;
  }
  

} //end MSP namespace




