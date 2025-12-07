
#include "modulepanel.h"


ModulePanel :: ModulePanel(wxWindow *parent,
			   wxWindowID id)
  : BasePanel(parent){

  this->Show(true);
  this->nameArray   = new wxArrayString();
  this->moduleArray = CreateArrayList(20);
  SetArrayListCleanFunc(this->moduleArray, NULL);

  //radio = new wxRadioBox(this, id, _T(""), wxDefaultPosition, wxDefaultSize, 0, NULL, 0, wxRA_SPECIFY_COLS, wxDefaultValidator, _T("radioBox"));
  radio = NULL;
  //this->sizer->Add(radio, 0, wxEXPAND);
  this->sizer->SetSizeHints(this);
  this->sizer->Layout();
}


ModulePanel :: ~ModulePanel(){
  delete this->nameArray;
  DestroyArrayList(&this->moduleArray);
}


void ModulePanel :: AddModule(Module *m){
  int id=-1;
  char name[512];
  m->GetName(name);
  wxString wxstr(name, wxConvUTF8);

  this->nameArray->Add(wxstr, 1);
  AddArrayListElement(this->moduleArray,(void *)m);

  if(radio!=NULL){
    id = radio->GetId();
    this->sizer->Detach(radio);
    delete radio;
  }

  radio = new wxRadioBox(this, id, _T(""), wxDefaultPosition, wxDefaultSize, *nameArray, 1, wxRA_SPECIFY_COLS, wxDefaultValidator, _T("radioBox"));
  this->sizer->Add(radio, 0, wxEXPAND);
  this->sizer->SetSizeHints(this);
  this->sizer->Layout();
}


Module * ModulePanel :: GetSelection(){
  if(radio==NULL)
    return NULL;
  int index = radio->GetSelection();
  if(index==wxNOT_FOUND)
    return NULL;
  return (Module *)GetArrayListElement(moduleArray, 
				       index);
}


