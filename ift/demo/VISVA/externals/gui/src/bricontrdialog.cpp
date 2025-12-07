
#include "bricontrdialog.h"


BriContrDialog::BriContrDialog(wxWindow *parent, wxWindowID id) 
  : wxDialog(parent, -1, _T(""), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("dialogBox")){
  
  vsizer = new wxBoxSizer(wxVERTICAL);
  sBrigh = new wxSlider(this, id, 50, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS, wxDefaultValidator, _T("slider0"));
  sContr = new wxSlider(this, id, 0, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS, wxDefaultValidator, _T("slider1"));

  wxStaticText *tBriContr = new wxStaticText(this, -1, _T("Brightness/Contrast"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE, _T("staticText1"));
    
  wxStaticLine *sline1 = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine1"));

  wxSizer *sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

  vsizer->AddSpacer(10);
  vsizer->Add(tBriContr,  0, wxALIGN_CENTER);
  vsizer->Add(sBrigh,  0, wxEXPAND);
  vsizer->Add(sContr,  0, wxEXPAND);
  vsizer->Add(sline1,  0, wxEXPAND);
  vsizer->Add(sbutton, 0, wxEXPAND);
  this->SetSizer(vsizer, true);
  vsizer->SetSizeHints(this);
  vsizer->Layout();

  Connect( wxID_CANCEL, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(BriContrDialog::OnCancel),
	   NULL, NULL );

  SetExtraStyle(~wxWS_EX_BLOCK_EVENTS);
}


BriContrDialog :: ~BriContrDialog(){
  Disconnect( wxID_CANCEL, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(BriContrDialog::OnCancel),
	      NULL, NULL );
}


void BriContrDialog::GetBCLevel(int *B, int *C){
  *C = sContr->GetValue();
  *B = sBrigh->GetValue();
}

void BriContrDialog::SetBCLevel(int B, int C){
  sContr->SetValue(C);
  sBrigh->SetValue(B);
}


void BriContrDialog::ShowWindow(){
  this->GetBCLevel(&B_old, &C_old);
  this->SetTitle(_T("Brightness & Contrast"));
  vsizer->SetSizeHints(this);
  vsizer->Layout();
  this->Show(true);
}


void BriContrDialog::OnCancel(wxCommandEvent& event){
  sBrigh->SetValue(B_old);
  sContr->SetValue(C_old);
  this->Show(false);

  int id = sContr->GetId();
  wxScrollEvent newevent( wxEVT_SCROLL_THUMBTRACK, id);
  newevent.SetEventObject( this );
  // Send it
  GetEventHandler()->ProcessEvent( newevent );
}


