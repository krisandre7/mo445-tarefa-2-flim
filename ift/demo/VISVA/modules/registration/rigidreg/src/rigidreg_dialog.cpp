
#include "rigidreg_dialog.h"

namespace RigidReg{

  RigidRegDialog::RigidRegDialog(Scene *scnR, Scene *scnG, Scene *scnB)
  //      : wxDialog(APP->Window, wxID_ANY, _T("Verify the registration"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("RigidRegDialog")){ 
    : wxDialog(APP->Window, wxID_ANY, _T("Verify the registration"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER , _T("RigidRegDialog")){ 

    wxSize size(300, 500);
    this->SetSize(size);
    this->SetMinSize(size);

    R=scnR;
    G=scnG;
    B=scnB;
    wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

    wxStaticText *help = new wxStaticText(this, wxID_ANY, _T("Notes: \n- Gray Voxels: good match \n- Green Voxels: fixed image (reference) \n- Red Voxels: moving image"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxStaticText *tmsg = new wxStaticText(this, wxID_ANY, _T("Orientation:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxStaticText *tmsg2 = new wxStaticText(this, wxID_ANY, _T("\nACCEPT THIS REGISTRATION?\n"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    int id_ori = APP->idManager->AllocID();
    wxString oriChoices[3];
    oriChoices[0] = _T("Coronal");
    oriChoices[1] = _T("Axial");
    oriChoices[2] = _T("Sagittal");
    chOri = new wxChoice(this, id_ori, wxDefaultPosition, wxDefaultSize, 3, oriChoices, 0, wxDefaultValidator, _T("choice0"));
    chOri->SetSelection(1);


    int id_spin = APP->idManager->AllocID();
    spinSlice = new wxSpinCtrl(this, id_spin, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS|wxSP_VERTICAL, 0, 0, 0, _T("ChangeSlices"));


    wxStaticLine *sline1 = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));

    wxStaticLine *sline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));
    wxSizer *sbutton = this->CreateButtonSizer(wxOK|wxCANCEL); 

    canvas = new Canvas(this);
    CImage *cimg = CreateCImage(R->zsize,R->xsize);
    Voxel u;
    int p,q;
    u.y = R->ysize/2;
    spinSlice->SetRange(0,R->ysize-1);
    spinSlice->SetValue(u.y);
    for (u.z=0; u.z < R->zsize; u.z++)
      for (u.x=0; u.x < R->xsize; u.x++) {
	p = u.x + R->tby[u.y] + R->tbz[u.z];
	q = u.z + cimg->C[0]->tbrow[u.x];
	cimg->C[0]->val[q] = (R->data[p]*255)/4096;
	cimg->C[1]->val[q] = (G->data[p]*255)/4096;
	cimg->C[2]->val[q] = (B->data[p]*255)/4096;
      }
    //WriteImage(cimg->C[0],"out.pgm");
    canvas->DrawCImage(cimg);
    DestroyCImage(&cimg);

    sizer->Add(tmsg,    0, wxEXPAND);
    sizer->Add(chOri,   0, wxEXPAND);
    sizer->Add(canvas,    1, wxEXPAND);
    sizer->Add(spinSlice,    0, wxALIGN_CENTER);
    sizer->Add(sline1,   0, wxEXPAND);
    sizer->Add(help,    0, wxEXPAND);
    sizer->Add(sline,   0, wxEXPAND);
    sizer->Add(tmsg2,    0, wxALIGN_CENTER);
    sizer->Add(sbutton, 0, wxEXPAND);

    this->SetSizer(sizer, true);
    sizer->SetSizeHints(this);
    sizer->Layout();

    Connect( id_ori, wxEVT_COMMAND_CHOICE_SELECTED,
	     wxCommandEventHandler(RigidRegDialog::OnChangeSliceOrientation),
	     NULL, NULL );
    Connect( id_spin, wxEVT_COMMAND_SPINCTRL_UPDATED,
	     wxCommandEventHandler(RigidRegDialog::OnChangeSpinCtrl),
	     NULL, NULL );


  }

  RigidRegDialog::~RigidRegDialog(){
    delete chOri;
  }

  
  int RigidRegDialog::GetOrientation(){
    return chOri->GetSelection()+1;
  }

  int RigidRegDialog::GetSpinValue(){
    return spinSlice->GetValue();
  }

  
  void RigidRegDialog::OnChangeSliceOrientation(wxCommandEvent& event){

    CImage *cimg;
    Voxel u;
    int p,q;
    if (this->GetOrientation()==1) {  //coronal
      cimg = CreateCImage(R->zsize,R->xsize);
      u.y = R->ysize/2;
      spinSlice->SetRange(0,R->ysize);
      spinSlice->SetValue(u.y);
      for (u.z=0; u.z < R->zsize; u.z++)
	for (u.x=0; u.x < R->xsize; u.x++) {
	  p = u.x + R->tby[u.y] + R->tbz[u.z];
	  q = u.z + cimg->C[0]->tbrow[u.x];
	  cimg->C[0]->val[q] = (R->data[p]*255)/4096;
	  cimg->C[1]->val[q] = (G->data[p]*255)/4096;
	  cimg->C[2]->val[q] = (B->data[p]*255)/4096;
	}
    }
    if (this->GetOrientation()==2) {  //axial
      cimg = CreateCImage(R->xsize,R->ysize);
      u.z = R->zsize/2;
      spinSlice->SetRange(0,R->zsize);
      spinSlice->SetValue(u.z);
      for (u.x=0; u.x < R->xsize; u.x++)
	for (u.y=0; u.y < R->ysize; u.y++) {
	  p = u.x + R->tby[u.y] + R->tbz[u.z];
	  q = u.x + cimg->C[0]->tbrow[u.y];
	  cimg->C[0]->val[q] = (R->data[p]*255)/4096;
	  cimg->C[1]->val[q] = (G->data[p]*255)/4096;
	  cimg->C[2]->val[q] = (B->data[p]*255)/4096;
	}
    }
    if (this->GetOrientation()==3) {  //sagittal
      cimg = CreateCImage(R->zsize,R->ysize);
      u.x = R->xsize/2;
      spinSlice->SetValue(u.x);
      spinSlice->SetRange(0,R->xsize);
      for (u.z=0; u.z < R->zsize; u.z++)
	for (u.y=0; u.y < R->ysize; u.y++) {
	  p = u.x + R->tby[u.y] + R->tbz[u.z];
	  q = u.z + cimg->C[0]->tbrow[u.y];
	  cimg->C[0]->val[q] = (R->data[p]*255)/4096;
	  cimg->C[1]->val[q] = (G->data[p]*255)/4096;
	  cimg->C[2]->val[q] = (B->data[p]*255)/4096;
	}
    }
    canvas->DrawCImage(cimg);
    canvas->Refresh();
    DestroyCImage(&cimg);


  }

  
  void RigidRegDialog::OnChangeSpinCtrl(wxCommandEvent& event){

    CImage *cimg;
    Voxel u;
    int p,q;
    if (this->GetOrientation()==1) {  //axial
      cimg = CreateCImage(R->zsize,R->xsize);
      u.y = this->GetSpinValue();
      for (u.z=0; u.z < R->zsize; u.z++)
	for (u.x=0; u.x < R->xsize; u.x++) {
	  p = u.x + R->tby[u.y] + R->tbz[u.z];
	  q = u.z + cimg->C[0]->tbrow[u.x];
	  cimg->C[0]->val[q] = (R->data[p]*255)/4096;
	  cimg->C[1]->val[q] = (G->data[p]*255)/4096;
	  cimg->C[2]->val[q] = (B->data[p]*255)/4096;
	}
    }
    if (this->GetOrientation()==2) {  //sagittal
      cimg = CreateCImage(R->xsize,R->ysize);
      u.z = this->GetSpinValue();
      for (u.x=0; u.x < R->xsize; u.x++)
	for (u.y=0; u.y < R->ysize; u.y++) {
	  p = u.x + R->tby[u.y] + R->tbz[u.z];
	  q = u.x + cimg->C[0]->tbrow[u.y];
	  cimg->C[0]->val[q] = (R->data[p]*255)/4096;
	  cimg->C[1]->val[q] = (G->data[p]*255)/4096;
	  cimg->C[2]->val[q] = (B->data[p]*255)/4096;
	}
    }
    if (this->GetOrientation()==3) {  //coronal
      cimg = CreateCImage(R->zsize,R->ysize);
      u.x = this->GetSpinValue();
      for (u.z=0; u.z < R->zsize; u.z++)
	for (u.y=0; u.y < R->ysize; u.y++) {
	  p = u.x + R->tby[u.y] + R->tbz[u.z];
	  q = u.z + cimg->C[0]->tbrow[u.y];
	  cimg->C[0]->val[q] = (R->data[p]*255)/4096;
	  cimg->C[1]->val[q] = (G->data[p]*255)/4096;
	  cimg->C[2]->val[q] = (B->data[p]*255)/4096;
	}
    }
    canvas->DrawCImage(cimg);
    canvas->Refresh();
    DestroyCImage(&cimg);
  }




} //end MSP namespace




