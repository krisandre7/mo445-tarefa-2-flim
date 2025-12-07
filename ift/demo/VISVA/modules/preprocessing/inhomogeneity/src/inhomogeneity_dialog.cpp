
#include "inhomogeneity_dialog.h"


namespace Inhomogeneity{

  InhomogeneityDialog::InhomogeneityDialog()
    : wxDialog(APP->Window, wxID_ANY, _T("Inhomogeneity correction settings"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("InhomogeneityDialog")){

    mainsizer = new wxBoxSizer(wxVERTICAL);
    sizer = new wxGridSizer(2,0,0);
    sizer2 = new wxGridSizer(2,0,0);

    tmsg0 = new wxStaticText(this, wxID_ANY, _T("Select Object:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    int i, choice = 0;
    int no = APP->segmobjs->n;
    wxString *objChoices;
    objChoices = new wxString[ no ];
    SegmObject *segm;
    for( i = 0; i < no; i++ ) {
      segm = ( SegmObject* ) GetArrayListElement( APP->segmobjs, i );
      objChoices[ i ] = wxString::FromAscii( segm->name );
      if( strcmp( segm->name, "Brain" ) == 0 )
	choice = i;
    }
    chObj = new wxChoice( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, no, ( wxString* ) objChoices, 0, wxDefaultValidator, _T( "choice0" ) );
    chObj->SetSelection( choice );
    
      
    //wxStaticText *tmsg = new wxStaticText(this, wxID_ANY, _T("Protocol:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    //wxString protoChoices[2];
    //protoChoices[0] = _T("MRI T1 (Default)");
    //protoChoices[1] = _T("MRI T2 or PD");
    //chProto = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 2, protoChoices, 0, wxDefaultValidator, _T("choice0"));
    //chProto->SetSelection(0);

    /*    
    wxStaticText *tmsg2 = new wxStaticText(this, wxID_ANY, _T("Compression:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxString compChoices[4];
    compChoices[0] = _T("1 - Best / Slower");
    compChoices[1] = _T("2 (Default)");
    compChoices[2] = _T("3");
    compChoices[3] = _T("4 - Worst / Faster");
    chComp = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 4, compChoices, 0, wxDefaultValidator, _T("choice0"));
    chComp->SetSelection(1);
    */


    sline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));

    sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

    int id_AdvOptions = APP->idManager->AllocID();
    adv_options = new wxCheckBox(this,id_AdvOptions,_T("Advanced Options:"),wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("check"));
    adv_options->SetValue(0);

    tmsg3 = new wxStaticText(this, wxID_ANY, _T("Function:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    func = new SpinFloat(this,wxID_ANY, 0, 4, 1.3, 0.1, 2);
    tmsg3->Hide();
    func->Hide();

    tmsg4 = new wxStaticText(this, wxID_ANY, _T("Radius(mm):"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    radius = new SpinFloat(this,wxID_ANY, 7, 28, 15.3, 0.1, 2);
    tmsg4->Hide();
    radius->Hide();

    //keep = new wxCheckBox(this,wxID_ANY,_T("Keep just the object voxels"),wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("check"));
    //keep->SetValue(0);
    

    sizer->Add(tmsg0,    0, wxEXPAND);
    sizer->Add(chObj,   0, wxEXPAND);
    //sizer->Add(tmsg,    0, wxEXPAND);
    //sizer->Add(chProto,   0, wxEXPAND);
    //sizer->Add(tmsg2,    0, wxEXPAND);
    //sizer->Add(chComp,   0, wxEXPAND);
    sizer2->Add(tmsg3,    0, wxEXPAND);
    sizer2->Add(func, 0, wxEXPAND);
    sizer2->Add(tmsg4,    0, wxEXPAND);
    sizer2->Add(radius, 0, wxEXPAND);

    mainsizer->Add(sizer, 0, wxEXPAND);
    mainsizer->Add(adv_options, 0, wxEXPAND);
    mainsizer->Add(sizer2, 0, wxEXPAND);
    //mainsizer->Add(keep, 0, wxEXPAND);
    mainsizer->Add(sline,   0, wxEXPAND);
    mainsizer->Add(sbutton, 0, wxEXPAND);
    

    this->SetSizer(mainsizer, true);
    mainsizer->SetSizeHints(this);
    mainsizer->Layout();

    Connect( id_AdvOptions, wxEVT_COMMAND_CHECKBOX_CLICKED,
             wxCommandEventHandler(InhomogeneityDialog::OnAdvOptions),
             NULL, NULL );
  }

  InhomogeneityDialog::~InhomogeneityDialog(){
    //delete chObj;
    //delete chProto;
    //delete chComp;
    delete func;
    delete radius;
    //delete keep;
  }

  
  //int InhomogeneityDialog::GetProtocol(){
  //return chProto->GetSelection();
  //}

  /*
  int InhomogeneityDialog::GetCompression(){
    return chComp->GetSelection()+1;
  }
  */

  Scene* InhomogeneityDialog::GetObject() {
    int p;
    char objname[1024];
    strcpy(objname,chObj->GetStringSelection().ToAscii());
    SegmObject *segm = APP->SearchObjByName(objname);
    Scene *scn = CreateScene(APP->Data.w,APP->Data.h,APP->Data.nframes);
    int n = APP->Data.w * APP->Data.h * APP->Data.nframes;
    for (p=0;p<n;p++) {
      if(_fast_BMapGet(segm->mask, p))
	scn->data[p] = 1;
    }
    return scn;
  }

  float InhomogeneityDialog::GetFunc(){
    return func->GetValue();
  }

  float InhomogeneityDialog::GetRadius(){
    return radius->GetValue();
  }

  /*
  int InhomogeneityDialog::GetKeep(){
    return (int)keep->GetValue();
  }
  */

  void InhomogeneityDialog::OnAdvOptions(wxCommandEvent& event) {
    if( adv_options->GetValue() == 1 ) {
      tmsg3->Show();
      func->Show();
      tmsg4->Show();
      radius->Show();
    }
    else {
      tmsg3->Hide();
      func->Hide();
      tmsg4->Hide();
      radius->Hide();
    }
    mainsizer->SetSizeHints(this);
  }

} //end Inhomogeneity namespace




