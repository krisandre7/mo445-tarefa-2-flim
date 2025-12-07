#include "braincluster_dialog.h"

namespace BrainCluster{

  BrainClusterDialog::BrainClusterDialog()
    : wxDialog(APP->Window, wxID_ANY, _T("Settings"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("BrainClusterDialog")){

    int i, default_obj = 0;
    int no = APP->segmobjs->n;
    wxString *objChoices;
    objChoices = new wxString[no];
    SegmObject *segm = NULL;
    float T_CSF_GM, T_GM_WM;
    Scene *mask = NULL;
    Scene *brain = NULL;
    for(i=0; i<no; i++){
      segm = (SegmObject *)GetArrayListElement(APP->segmobjs, i);
      objChoices[i] = wxString::FromAscii(segm->name);
      if( strcmp( segm->name, "Brain" ) == 0 )
	default_obj = i;
    }

    mask = CreateScene( APP->Data.w, APP->Data.h, APP->Data.nframes );
    CopySegmObjectMask2Scene( segm, mask );
    brain = Mult3( APP->Data.orig, mask );
    BrainPreProportions( brain, mask, ( int ) APP->Data.modality, &T_CSF_GM, &T_GM_WM );
    DestroyScene( &brain );
    DestroyScene( &mask );

    int id_Obj = APP->idManager->AllocID();
    txtChObj = new wxStaticText(this, wxID_ANY, _T("Select Object:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    chObj = new wxChoice(this, id_Obj, wxDefaultPosition, wxDefaultSize, no, (wxString *) objChoices, 0, wxDefaultValidator, _T("choice0"));
    chObj->SetSelection(default_obj);

    int id_AdvOptions = APP->idManager->AllocID();
    adv_options = new wxCheckBox(this,id_AdvOptions,_T("Advanced Options:"),wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("check"));
    adv_options->SetValue(0);

    txtSamples = new wxStaticText(this, wxID_ANY, _T("WM/GM Samples:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    samples = new SpinFloat(this,wxID_ANY, 300, 2000, 1000, 1, 0);
    txtSamples->Hide();
    samples->Hide();

    txtSamplesCSF = new wxStaticText(this, wxID_ANY, _T("CSF/WMGM Samples:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    samplesCSF = new SpinFloat(this,wxID_ANY, 300, 2000, 1000, 1, 0);
    txtSamplesCSF->Hide();
    samplesCSF->Hide();

    txtMean_Prop = new wxStaticText(this, wxID_ANY, _T("WM/GM Mean proportion:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    mean_prop = new SpinFloat(this,wxID_ANY, 0.0, 1.0, T_GM_WM , 0.05, 2);
    mean_prop->Hide();
    txtMean_Prop->Hide();

    int id_Auto = APP->idManager->AllocID();
    auto_prop = new wxCheckBox(this,id_Auto,_T("WM/GM Automatic tissue proportions"),wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("check"));
    auto_prop->SetValue(1);
    auto_prop->Hide();
    
    txtKMin = new wxStaticText(this, wxID_ANY, _T("WM/GM K-nn minimum:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    Kmin = new SpinFloat(this,wxID_ANY, 10, 60, 10, 1.0, 0);
    Kmin->Hide();
    txtKMin->Hide();

    txtKMax = new wxStaticText(this, wxID_ANY, _T("WM/GM K-nn maximum:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    Kmax = new SpinFloat(this,wxID_ANY, 10, 60, 20, 1.0, 0);
    Kmax->Hide();
    txtKMax->Hide();

    txtMean_PropCSF = new wxStaticText(this, wxID_ANY, _T("CSF/WMGM Mean proportion:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    mean_propcsf = new SpinFloat(this,wxID_ANY, 0.0, 1.0, T_CSF_GM, 0.05, 2);
    mean_propcsf->Hide();
    txtMean_PropCSF->Hide();

    int id_AutoCSF = APP->idManager->AllocID();
    auto_propcsf = new wxCheckBox(this,id_AutoCSF,_T("CSF/WMGM Automatic tissue proportions"),wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("check"));
    auto_propcsf->SetValue(1);
    auto_propcsf->Hide();
    
    txtKMinCSF = new wxStaticText(this, wxID_ANY, _T("CSF/WMGM K-nn minimum:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    Kmincsf = new SpinFloat(this,wxID_ANY, 10, 60, 10, 1.0, 0);
    Kmincsf->Hide();
    txtKMinCSF->Hide();

    txtKMaxCSF = new wxStaticText(this, wxID_ANY, _T("CSF/WMGM K-nn maximum:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    Kmaxcsf = new SpinFloat(this,wxID_ANY, 10, 60, 20, 1.0, 0);
    Kmaxcsf->Hide();
    txtKMaxCSF->Hide();

    sline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));

    // Color buttons
    txtCSF = new wxStaticText(this, wxID_ANY, _T("Color for CSF:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    txtWM = new wxStaticText(this, wxID_ANY, _T("Color for WM:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    txtGM = new wxStaticText(this, wxID_ANY, _T("Color for GM:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    butColorCSF = new ColourButton(this, wxID_ANY);
    butColorWM = new ColourButton(this, wxID_ANY);
    butColorGM = new ColourButton(this, wxID_ANY);
    butColorCSF->SetValue(0x0000FF);
    butColorWM->SetValue(0xFFA500);
    butColorGM->SetValue(0xADD8E6);
    int id_csfCheck = APP->idManager->AllocID();
    csfCheck = new wxCheckBox(this,id_csfCheck,_T("Segment CSF"),wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("check"));
    csfCheck->SetValue(1);

    mainsizer = new wxBoxSizer(wxVERTICAL);
    sizer = new wxGridSizer(2,0,0);
    sizer2 = new wxGridSizer(2,0,0);
    sizer3 = new wxGridSizer(4,0,0);
    sizer4 = new wxBoxSizer(wxVERTICAL);
    sizer5 = new wxGridSizer(4,0,0);
    sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

    sizer->Add(txtChObj,    0, wxEXPAND);
    sizer->Add(chObj,   0, wxEXPAND);
    sizer->Add(txtWM,    0, wxEXPAND);
    sizer->Add(butColorWM,   0, wxEXPAND);
    sizer->Add(txtGM,    0, wxEXPAND);
    sizer->Add(butColorGM,   0, wxEXPAND);
    sizer->Add(csfCheck,   0, wxEXPAND);
    sizer->Add(new wxStaticText(this, -1, wxT("")), 0, wxEXPAND);
    sizer->Add(txtCSF,    0, wxEXPAND);
    sizer->Add(butColorCSF,   0, wxEXPAND);
    sizer2->Add(auto_prop,   0, wxEXPAND);
    sizer2->Add(auto_propcsf,   0, wxEXPAND);
    sizer3->Add(txtSamples,    0, wxEXPAND);
    sizer3->Add(samples,   0, wxEXPAND);
    sizer3->Add(txtSamplesCSF,    0, wxEXPAND);
    sizer3->Add(samplesCSF,   0, wxEXPAND);
    sizer3->Add(txtKMin,    0, wxEXPAND);
    sizer3->Add(Kmin,   0, wxEXPAND);
    sizer3->Add(txtKMinCSF,    0, wxEXPAND);
    sizer3->Add(Kmincsf,   0, wxEXPAND);
    sizer3->Add(txtKMax,    0, wxEXPAND);
    sizer3->Add(Kmax,   0, wxEXPAND);
    sizer3->Add(txtKMaxCSF,    0, wxEXPAND);
    sizer3->Add(Kmaxcsf,   0, wxEXPAND);
    sizer5->Add(txtMean_Prop,    0, wxEXPAND);
    sizer5->Add(mean_prop,   0, wxEXPAND);
    sizer5->Add(txtMean_PropCSF,    0, wxEXPAND);
    sizer5->Add(mean_propcsf,   0, wxEXPAND);
    sizer4->Add(sline,   0, wxEXPAND);
    mainsizer->Add(sizer, 0, wxEXPAND);
    mainsizer->Add(adv_options, 0, wxEXPAND);
    mainsizer->Add(sizer3, 0, wxEXPAND);
    mainsizer->Add(sizer2, 0, wxEXPAND);
    mainsizer->Add(sizer5, 0, wxEXPAND);
    mainsizer->Add(sizer4,   0, wxEXPAND);
    mainsizer->Add(sbutton, 0, wxEXPAND);
    
    this->SetSizer(mainsizer, true);
    mainsizer->SetSizeHints(this);
    mainsizer->Layout();

    Connect( id_Obj, wxEVT_COMMAND_CHOICE_SELECTED,
             wxCommandEventHandler(BrainClusterDialog::OnChangeObject),
             NULL, NULL );
    Connect( id_Auto, wxEVT_COMMAND_CHECKBOX_CLICKED,
             wxCommandEventHandler(BrainClusterDialog::OnChangeCheckBox),
             NULL, NULL );
    Connect( id_AdvOptions, wxEVT_COMMAND_CHECKBOX_CLICKED,
             wxCommandEventHandler(BrainClusterDialog::OnAdvOptions),
             NULL, NULL );
    Connect( id_AutoCSF, wxEVT_COMMAND_CHECKBOX_CLICKED,
             wxCommandEventHandler(BrainClusterDialog::OnChangeCheckBoxCSF),
             NULL, NULL );
    Connect( id_csfCheck, wxEVT_COMMAND_CHECKBOX_CLICKED,
             wxCommandEventHandler(BrainClusterDialog::OnCsfCheck),
             NULL, NULL );


  }

  BrainClusterDialog::~BrainClusterDialog(){
    //delete chProto;
    //delete chComp;
  }

  
  Scene* BrainClusterDialog::GetObject() {
    int p;
    char objname[1024];
    strcpy(objname,chObj->GetStringSelection().ToAscii());
    SegmObject *segm = APP->SearchObjByName(objname);
    Scene *scn = CreateScene(APP->Data.w,APP->Data.h,APP->Data.nframes);
    int n = APP->Data.w * APP->Data.h * APP->Data.nframes;
    for (p=0;p<n;p++) 
    if(_fast_BMapGet(segm->mask, p))
    scn->data[p] = 1;
    return scn;
    //return NULL;
  }
  
  float BrainClusterDialog::GetSamples() {
    return samples->GetValue();
  }

  float BrainClusterDialog::GetSamplesCSF() {
    return samplesCSF->GetValue();
  }
  
  float BrainClusterDialog::GetMean_Prop() {
    return mean_prop->GetValue();
  }
  
  float BrainClusterDialog::GetMean_PropCSF() {
    return mean_propcsf->GetValue();
  }
  
  float BrainClusterDialog::GetAuto_Prop() {
    return (float)auto_prop->GetValue();
  }	
  
  float BrainClusterDialog::GetKMin() {
    return Kmin->GetValue();
  }	
  
  float BrainClusterDialog::GetKMax() {
    return Kmax->GetValue();
  }	
  
  float BrainClusterDialog::GetAuto_PropCSF() {
    return (float)auto_propcsf->GetValue();
  }	
  
  float BrainClusterDialog::GetKMinCSF() {
    return Kmincsf->GetValue();
  }	
  
  float BrainClusterDialog::GetKMaxCSF() {
    return Kmaxcsf->GetValue();
  }	
  
  int BrainClusterDialog::GetCsfCheck() {
    return csfCheck->GetValue();
  }	


  int BrainClusterDialog::GetColorCSF() {
    return butColorCSF->GetValue();
  }	

  int BrainClusterDialog::GetColorWM() {
    return butColorWM->GetValue();
  }	

  int BrainClusterDialog::GetColorGM() {
    return butColorGM->GetValue();
  }	
  
  void BrainClusterDialog::RefreshDialog() {
    txtChObj->Hide();
    chObj->Hide();
    mainsizer->SetSizeHints(this);
    txtChObj->Show();
    chObj->Show();
    mainsizer->SetSizeHints(this);
  }

  void BrainClusterDialog::OnChangeObject( wxCommandEvent& event ) {
    float T_CSF_GM;
    float T_GM_WM;
    Scene *mask;
    Scene *brain = Mult3( APP->Data.orig, mask );

    if( csfCheck->GetValue() == 1 ) {
      BrainPreProportions( brain, mask, ( int ) APP->Data.modality, &T_CSF_GM, &T_GM_WM );
      mean_propcsf->SetValue( T_CSF_GM );
    }
    else
      TissueProportions( brain, mask, ( int ) APP->Data.modality, WMGM_SEG, 1.0, 0.0, &T_GM_WM );
    mean_prop->SetValue( T_GM_WM );

    DestroyScene( &brain );
    DestroyScene( &mask );
    this->RefreshDialog();
  }

  void BrainClusterDialog::OnChangeCheckBox( wxCommandEvent& event ) {
    this->SetCheckBox();
    //this->RefreshDialog();
  }

  void BrainClusterDialog::SetCheckBox(){
    if (auto_prop->GetValue()==0) {
      txtMean_Prop->Enable();
      mean_prop->Enable();
      // txtKMin->Enable();
      // Kmin->Enable();
      // txtKMax->Enable();
      // Kmax->Enable();
    }
    else {
      txtMean_Prop->Disable();
      mean_prop->Disable();
      // txtKMin->Disable();
      // Kmin->Disable();
      // txtKMax->Disable();
      // Kmax->Disable();
    }
  }

  void BrainClusterDialog::OnChangeCheckBoxCSF(wxCommandEvent& event){
    this->SetCheckBoxCSF();
    //this->RefreshDialog();
  }
  
  void BrainClusterDialog::SetCheckBoxCSF(){
    if( auto_propcsf->GetValue()==0 ) {
      txtMean_PropCSF->Enable();
      mean_propcsf->Enable();
      // txtKMinCSF->Enable();
      // Kmincsf->Enable();
      // txtKMaxCSF->Enable();
      // Kmaxcsf->Enable();
    }
    else {
      txtMean_PropCSF->Disable();
      mean_propcsf->Disable();
      // txtKMinCSF->Disable();
      // Kmincsf->Disable();
      // txtKMaxCSF->Disable();
      // Kmaxcsf->Disable();
    }
  }

  void BrainClusterDialog::OnCsfCheck(wxCommandEvent& event){
    this->SetCsf();
    //this->RefreshDialog();
  }
  
  void BrainClusterDialog::SetCsf(){
    float T_CSF_GM;
    float T_GM_WM;
    char objname[ 1024 ];
    SegmObject *segm;
    Scene *brain;
    Scene *mask;
    
    strcpy( objname, chObj->GetStringSelection( ).ToAscii( ) );
    segm = APP->SearchObjByName( objname );
    mask = CreateScene( APP->Data.w, APP->Data.h, APP->Data.nframes );
    CopySegmObjectMask2Scene( segm, mask );
    brain = Mult3( APP->Data.orig, mask );
    if( csfCheck->GetValue() == 1 ) {
      // Computing WMGM proportion over WMGMCSF mask.
      BrainPreProportions( brain, mask, ( int ) APP->Data.modality, &T_CSF_GM, &T_GM_WM );
      mean_prop->SetValue( T_GM_WM );
      mean_propcsf->SetValue( T_CSF_GM );

      butColorCSF->Enable();
      txtCSF->Enable();
      txtSamplesCSF->Enable();
      samplesCSF->Enable();
      txtKMinCSF->Enable();
      Kmincsf->Enable();
      txtKMaxCSF->Enable();
      Kmaxcsf->Enable();
      if( adv_options->GetValue() == 1 ) {
	auto_propcsf->Enable();
	if( auto_propcsf->GetValue() == 1 ) {
	  txtMean_PropCSF->Disable();
	  mean_propcsf->Disable();
	}
	else {
	  txtMean_PropCSF->Enable();
	  mean_propcsf->Enable();
	}
      }
    }
    else {
      // Computing WMGM proportion over WMGM mask.
      TissueProportions( brain, mask, ( int ) APP->Data.modality, WMGM_SEG, 1.0, 0.0, &T_GM_WM );
      mean_prop->SetValue( T_GM_WM );
   
      butColorCSF->Disable();
      txtCSF->Disable();
      txtSamplesCSF->Disable();
      samplesCSF->Disable();
      txtKMinCSF->Disable();
      Kmincsf->Disable();
      txtKMaxCSF->Disable();
      Kmaxcsf->Disable();
      if( adv_options->GetValue() == 1 ) {
	auto_propcsf->Disable();
	txtMean_PropCSF->Disable();
	mean_propcsf->Disable();
      }
    }
    DestroyScene( &brain );
  }
  
  void BrainClusterDialog::OnAdvOptions(wxCommandEvent& event){
    if( adv_options->GetValue() == 1 ) {
      txtSamples->Show();
      samples->Show();
      txtSamplesCSF->Show();
      samplesCSF->Show();
      auto_prop->Show();
      auto_propcsf->Show();
      // WMGM options
      txtMean_Prop->Show();
      mean_prop->Show();
      txtKMin->Show();
      Kmin->Show();
      txtKMax->Show();
      Kmax->Show();
      // CSF options
      //csfCheck->Show();
      txtMean_PropCSF->Show();
      mean_propcsf->Show();
      txtKMinCSF->Show();
      Kmincsf->Show();
      txtKMaxCSF->Show();
      Kmaxcsf->Show();
      this->SetCsf();
      this->SetCheckBox();
      if( csfCheck->GetValue()==1 )
	this->SetCheckBoxCSF();
    }
    else {
      txtSamples->Hide();
      samples->Hide();
      txtSamplesCSF->Hide();
      samplesCSF->Hide();
      auto_prop->Hide();
      auto_propcsf->Hide();
      // WMGM options
      txtMean_Prop->Hide();
      mean_prop->Hide();
      txtKMin->Hide();
      Kmin->Hide();
      txtKMax->Hide();
      Kmax->Hide();
      // CSF options
      //csfCheck->Hide();
      txtMean_PropCSF->Hide();
      mean_propcsf->Hide();
      txtKMinCSF->Hide();
      Kmincsf->Hide();
      txtKMaxCSF->Hide();
      Kmaxcsf->Hide();
    }
    this->RefreshDialog();
  }
} //end BrainCluster namespace




