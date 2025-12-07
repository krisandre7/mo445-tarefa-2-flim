
#include "basedialog.h"


BaseDialog :: BaseDialog(wxWindow *parent,
			 char     *title) 
  : wxDialog(parent, -1, _T(""), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("dialogBox")){
  this->parent = parent;
  strcpy(this->title, title);
  //SetBackgroundColour(*wxWHITE);
  wxString wxtitle(title, wxConvUTF8);
  this->SetTitle(wxtitle);

  sizer = new wxBoxSizer(wxVERTICAL);

  wxStaticLine *sline1 = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, 
					  wxDefaultSize, wxLI_HORIZONTAL, 
					  _T("staticLine"));
  wxSizer *sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

  sizer->Add(sline1,  0, wxEXPAND);
  sizer->Add(sbutton, 0, wxEXPAND);

  this->SetSizer(sizer, true);
  sizer->SetSizeHints(this);
  sizer->Layout();

  Connect( wxID_CANCEL, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(BaseDialog::OnCancel),
	   NULL, NULL );
  Connect( wxID_OK, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(BaseDialog::OnOk),
	   NULL, NULL );
}


BaseDialog :: ~BaseDialog(){
  Disconnect( wxID_CANCEL, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(BaseDialog::OnCancel),
	      NULL, NULL );
  Disconnect( wxID_OK, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(BaseDialog::OnOk),
	      NULL, NULL );
  //sizer->Clear(true);
}


void BaseDialog::AddPanel(wxPanel *panel){
  sizer->Prepend(panel, 1, wxEXPAND);
  //sizer->PrependSpacer(10);

  this->SetSizer(sizer, true);
  sizer->SetSizeHints(this);
  sizer->Layout();
}

void BaseDialog::AddPanel(wxPanel *panel, int proportion){
  sizer->Prepend(panel, proportion, wxEXPAND);
  //sizer->PrependSpacer(10);

  this->SetSizer(sizer, true);
  sizer->SetSizeHints(this);
  sizer->Layout();
}


void BaseDialog::DetachPanel(wxPanel *panel){
  sizer->Detach(panel);
}


void BaseDialog::ShowWindow(){
  if(this->IsModal())
    this->EndModal(wxID_CANCEL);
  else
    this->Show(false);

  sizer->SetSizeHints(this);
  sizer->Layout();
  this->Show(true);
}

void BaseDialog::OnCancel(wxCommandEvent& event){
  if(this->IsModal())
    this->EndModal(wxID_CANCEL);
  this->Show(false);
}


void BaseDialog::OnOk(wxCommandEvent& event){
  if(this->IsModal())
    this->EndModal(wxID_OK);
  this->Show(false);
}


int  BaseDialog::ShowModal(){
  if(this->IsModal())
    this->EndModal(wxID_CANCEL);
  else
    this->Show(false);

  wxWindowDisabler disableAll(this);
  wxTheApp->Yield();

  //sizer->SetSizeHints(this);
  //sizer->Layout();

  return wxDialog::ShowModal();
}

