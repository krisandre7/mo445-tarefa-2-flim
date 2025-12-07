
#include "msp_dialog.h"

namespace MSP{

  MSPDialog::MSPDialog()
    : wxDialog(APP->Window, wxID_ANY, _T("Settings"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("MSPDialog")){

    wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

    wxStaticText *tmsg = new wxStaticText(this, wxID_ANY, _T("Choose the current volume orientation:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxString oriChoices[4];
    oriChoices[0] = _T("Auto-detect");
    oriChoices[1] = _T("Sagittal");
    oriChoices[2] = _T("Axial");
    oriChoices[3] = _T("Coronal");
    chOri = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 4, oriChoices, 0, wxDefaultValidator, _T("choice0"));
    if(APP->Data.oriented == 1 ) 
      chOri->SetSelection(2);
    else
      chOri->SetSelection(0);

    wxStaticText *tmsg2 = new wxStaticText(this, wxID_ANY, _T("Choose the accuracy/speed:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxString acuChoices[3];
    acuChoices[0] = _T("Best Accuracy (Slowest)");
    acuChoices[1] = _T("Good Accuracy (Moderate)");
    acuChoices[2] = _T("Fair Accuracy (Fastest)");
    acuOri = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 3, acuChoices, 0, wxDefaultValidator, _T("choice0"));
    acuOri->SetSelection(0);


    wxStaticText *tmsg3 = new wxStaticText(this, wxID_ANY, _T("Object to consider:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    int i;
    int no = APP->segmobjs->n;
    wxString *objChoices;
    objChoices = new wxString[no+1];
    SegmObject *segm;
    objChoices[0] = wxString::FromAscii("Whole Image (default)");
    for(i=0; i<no; i++){
      segm = (SegmObject *)GetArrayListElement(APP->segmobjs, i);
      objChoices[i+1] = wxString::FromAscii(segm->name);
    }
    chObj = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, no+1, (wxString *) objChoices, 0, wxDefaultValidator, _T("choice0"));
    chObj->SetSelection(0);


    wxStaticLine *sline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));
    wxSizer *sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

    sizer->AddSpacer(15);
    sizer->Add(tmsg,    0, wxEXPAND);
    sizer->Add(chOri,   0, wxEXPAND);
    sizer->AddSpacer(15);
    sizer->Add(tmsg2,    0, wxEXPAND);
    sizer->Add(acuOri,   0, wxEXPAND);
    sizer->AddSpacer(15);
    sizer->Add(tmsg3,    0, wxEXPAND);
    sizer->Add(chObj,   0, wxEXPAND);
    sizer->AddSpacer(15);
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
    return chOri->GetSelection();
  }

  int MSPDialog::GetAccuracy(){
    return acuOri->GetSelection()+1;
  }


  Scene* MSPDialog::GetObject() {
    int p;
    char objname[1024];
    strcpy(objname,chObj->GetStringSelection().ToAscii());
    if (strcmp(objname,"Whole Image (default)")==0) return NULL;
    SegmObject *segm = APP->SearchObjByName(objname);
    Scene *scn = CreateScene(APP->Data.w,APP->Data.h,APP->Data.nframes);
    int n = APP->Data.w * APP->Data.h * APP->Data.nframes;
    for (p=0;p<n;p++) 
      if(_fast_BMapGet(segm->mask, p))
	scn->data[p] = 1;
    return scn;
  }

  

} //end MSP namespace




