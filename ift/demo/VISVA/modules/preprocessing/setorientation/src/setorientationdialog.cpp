
#include "setorientationdialog.h"
#include "arrow.xpm"

namespace Setorientation{

  SetorientationDialog::SetorientationDialog()
    : wxDialog(APP->Window, wxID_ANY, _T("Set Image Orientation"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("SetorientationDialog")){

    Scene *scn = APP->Data.orig;
    scn->maxval = MaximumValue3(scn);

    wxStaticText *initmsg = new wxStaticText(this, wxID_ANY, _T("\nSELECT WHAT IS SEEN ON TOP OF EACH SLICE:\n(Note: Left/right is automaticaly assigned, but can be changed in case of flipped images).\n"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    wxString choices[7];
    choices[0] = _T("---");
    choices[1] = _T("Superior (Top)");
    choices[2] = _T("Inferior (Bottom)");
    choices[3] = _T("Anterior (Front)");
    choices[4] = _T("Posterior (Back)");
    choices[5] = _T("Left");
    choices[6] = _T("Right");
    int id_choiceX = APP->idManager->AllocID();
    int id_choiceY = APP->idManager->AllocID();
    int id_choiceZ = APP->idManager->AllocID();
    choiceX = new wxChoice(this, id_choiceX, wxDefaultPosition, wxDefaultSize, 7, choices, 0, wxDefaultValidator, _T("choiceX"));
    choiceY = new wxChoice(this, id_choiceY, wxDefaultPosition, wxDefaultSize, 7, choices, 0, wxDefaultValidator, _T("choiceY"));
    choiceZ = new wxChoice(this, id_choiceZ, wxDefaultPosition, wxDefaultSize, 7, choices, 0, wxDefaultValidator, _T("choiceZ"));
    choiceX->SetSelection(0);
    choiceY->SetSelection(0);
    choiceZ->SetSelection(0);

    wxBitmap bitmap(arrow_xpm);
    wxStaticBitmap *arrowX = new wxStaticBitmap(this, -1, bitmap);
    wxStaticBitmap *arrowY = new wxStaticBitmap(this, -1, bitmap);
    wxStaticBitmap *arrowZ = new wxStaticBitmap(this, -1, bitmap);

    int id_spinX = APP->idManager->AllocID();
    int id_spinY = APP->idManager->AllocID();
    int id_spinZ = APP->idManager->AllocID();
    spinSliceX = new wxSpinCtrl(this, id_spinX, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS|wxSP_VERTICAL, 0, 0, 0, _T("ChangeSlicesX"));
    spinSliceY = new wxSpinCtrl(this, id_spinY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS|wxSP_VERTICAL, 0, 0, 0, _T("ChangeSlicesY"));
    spinSliceZ = new wxSpinCtrl(this, id_spinZ, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS|wxSP_VERTICAL, 0, 0, 0, _T("ChangeSlicesZ"));
    spinSliceX->SetRange(0,scn->xsize-1);
    spinSliceY->SetRange(0,scn->ysize-1);
    spinSliceZ->SetRange(0,scn->zsize-1);
    spinSliceX->SetValue(scn->xsize/2);
    spinSliceY->SetValue(scn->ysize/2);
    spinSliceZ->SetValue(scn->zsize/2);
 
    canvasX = new Canvas(this);
    canvasY = new Canvas(this);
    canvasZ = new Canvas(this);
    UpdateCanvas();

    wxStaticLine *sline1 = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));
    wxStaticLine *sline2 = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine"));

    wxSizer *sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

    /*
    wxBoxSizer *canvassizer1 = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer *canvassizer2 = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer *canvassizer3 = new wxBoxSizer(wxVERTICAL);
    canvassizer1->Add(canvasX, 1, wxEXPAND);
    canvassizer2->Add(canvasY, 1, wxEXPAND);
    canvassizer3->Add(canvasZ, 1, wxEXPAND);
    canvassizer1->SetMin
    */

    wxBoxSizer *mainsizer = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer *vertsizer1 = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer *vertsizer2 = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer *vertsizer3 = new wxBoxSizer(wxVERTICAL);
    wxFlexGridSizer *horizsizer = new wxFlexGridSizer(3,0,10); //wxHORIZONTAL);

    vertsizer1->Add(choiceX, 0, wxALIGN_CENTER);
    vertsizer1->Add(arrowX, 0, wxALIGN_CENTER);
    vertsizer1->Add(canvasX, 1, wxEXPAND);
    vertsizer1->Add(spinSliceX, 0, wxALIGN_CENTER);

    vertsizer2->Add(choiceY, 0, wxALIGN_CENTER);
    vertsizer2->Add(arrowY, 0, wxALIGN_CENTER);
    vertsizer2->Add(canvasY, 1, wxEXPAND);
    vertsizer2->Add(spinSliceY, 0, wxALIGN_CENTER);

    vertsizer3->Add(choiceZ, 0, wxALIGN_CENTER);
    vertsizer3->Add(arrowZ, 0, wxALIGN_CENTER);
    vertsizer3->Add(canvasZ, 1, wxEXPAND);
    vertsizer3->Add(spinSliceZ, 0, wxALIGN_CENTER);

    vertsizer1->SetItemMinSize(canvasX,256,256);
    vertsizer2->SetItemMinSize(canvasY,256,256);
    vertsizer3->SetItemMinSize(canvasZ,256,256);

    horizsizer->Add(vertsizer1, 1, wxEXPAND);
    horizsizer->Add(vertsizer2, 1, wxEXPAND);
    horizsizer->Add(vertsizer3, 1, wxEXPAND);

    mainsizer->Add(initmsg, 0, wxEXPAND);
    mainsizer->Add(sline1, 0, wxEXPAND);
    mainsizer->Add(horizsizer, 1, wxEXPAND);
    mainsizer->Add(sline2, 0, wxEXPAND);
    mainsizer->Add(sbutton, 0, wxEXPAND);

    this->SetSizer(mainsizer, true);
    mainsizer->SetSizeHints(this);
    mainsizer->Layout();

    Connect( id_spinX, wxEVT_COMMAND_SPINCTRL_UPDATED,
	     wxCommandEventHandler(SetorientationDialog::OnChangeSpinCtrl),
	     NULL, NULL );
    Connect( id_spinY, wxEVT_COMMAND_SPINCTRL_UPDATED,
	     wxCommandEventHandler(SetorientationDialog::OnChangeSpinCtrl),
	     NULL, NULL );
    Connect( id_spinZ, wxEVT_COMMAND_SPINCTRL_UPDATED,
	     wxCommandEventHandler(SetorientationDialog::OnChangeSpinCtrl),
	     NULL, NULL );

    Connect( id_choiceX, wxEVT_COMMAND_CHOICE_SELECTED,
	     wxCommandEventHandler(SetorientationDialog::OnChangeChoiceX),
	     NULL, NULL );
    Connect( id_choiceY, wxEVT_COMMAND_CHOICE_SELECTED,
	     wxCommandEventHandler(SetorientationDialog::OnChangeChoiceY),
	     NULL, NULL );
    Connect( id_choiceZ, wxEVT_COMMAND_CHOICE_SELECTED,
	     wxCommandEventHandler(SetorientationDialog::OnChangeChoiceZ),
	     NULL, NULL );

  }

  SetorientationDialog::~SetorientationDialog(){
    /* delete sdx;
    delete sdy;
    delete sdz;
    */
  }

  void SetorientationDialog::UpdateCanvas(){

    CImage *cimg;
    Image *img;
    Scene *scn = APP->Data.orig;
    int maxval = scn->maxval;

    // plane X
    int i,j;
    Voxel v;
    v.x = spinSliceX->GetValue();
    img = CreateImage(scn->zsize,scn->ysize);
    for (v.z=0;v.z<scn->zsize;v.z++)
      for (v.y=0;v.y<scn->ysize;v.y++) {
	i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	if (scn->data[i] != 0) {
	  j = img->tbrow[v.y] + v.z;
	  img->val[j] = scn->data[i];
	}
      }
    LinearStretchinplace(img,0,(int)(maxval*0.9),0,255);
    cimg = Convert2CImage(img);
    DestroyImage(&img);
    canvasX->DrawCImage(cimg);
    canvasX->Refresh();
    DestroyCImage(&cimg);

    // plane Y
    img = GetYSlice(scn,spinSliceY->GetValue());
    LinearStretchinplace(img,0,(int)(maxval*0.9),0,255);
    cimg = Convert2CImage(img);
    DestroyImage(&img);
    canvasY->DrawCImage(cimg);
    canvasY->Refresh();
    DestroyCImage(&cimg);

    // plane Z
    v.z = spinSliceZ->GetValue();
    img = CreateImage(scn->ysize,scn->xsize);
    for (v.x=0;v.x<scn->xsize;v.x++)
      for (v.y=0;v.y<scn->ysize;v.y++) {
	i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	if (scn->data[i] != 0) {
	  j = img->tbrow[v.x] + v.y;
	  img->val[j] = scn->data[i];
	}
      }
    LinearStretchinplace(img,0,(int)(maxval*0.9),0,255);
    cimg = Convert2CImage(img);
    DestroyImage(&img);
    canvasZ->DrawCImage(cimg);
    canvasZ->Refresh();
    DestroyCImage(&cimg);
  }


  void SetorientationDialog::OnChangeSpinCtrl(wxCommandEvent& event){
    UpdateCanvas();
  }


  void SetorientationDialog::OnChangeChoiceX(wxCommandEvent& event){
    int x = this->GetChoiceX();
    int y = this->GetChoiceY();
    int z = this->GetChoiceZ();
    int typex,typey,typez;
    typex = (x+1)/2;  // sup/inf (type=1), ant/pos (type=2), left/right (type=3)
    typey = (y+1)/2;  // not selected (type=0)
    typez = (z+1)/2;
    if (typey==0 && typez==0) return; // not selected
    if (typex==typey) {
      choiceY->SetSelection(0);
      if (typez==3) choiceZ->SetSelection(0);
      return;
    }
    if (typex==typez) {
      choiceZ->SetSelection(0);
      if (typey==3) choiceY->SetSelection(0);
      return;
    }
    // autosetup left/right
    if (typex!=3) AutoLeftRight();
  }



  void SetorientationDialog::OnChangeChoiceY(wxCommandEvent& event){
    int x = this->GetChoiceX();
    int y = this->GetChoiceY();
    int z = this->GetChoiceZ();
    int typex,typey,typez;
    typex = (x+1)/2;  // sup/inf (type=1), ant/pos (type=2) ...
    typey = (y+1)/2;
    typez = (z+1)/2;
    if (typex==0 && typez==0) return; // not selected
    if (typey==typex) {
      choiceX->SetSelection(0);
      if (typez==3) choiceZ->SetSelection(0);
      return;
    }
    if (typey==typez) {
      choiceZ->SetSelection(0);
      if (typex==3) choiceX->SetSelection(0);
      return;
    }
    // autosetup left/right
    if (typey!=3) AutoLeftRight();
  }

  void SetorientationDialog::OnChangeChoiceZ(wxCommandEvent& event){
    int x = this->GetChoiceX();
    int y = this->GetChoiceY();
    int z = this->GetChoiceZ();
    int typex,typey,typez;
    typex = (x+1)/2;  // sup/inf (type=1), ant/pos (type=2) ...
    typey = (y+1)/2;
    typez = (z+1)/2;
    if (typex==0 && typey==0) return; // not selected
    if (typez==typex) {
      choiceX->SetSelection(0);
      if (typey==3) choiceY->SetSelection(0);
      return;
    }
    if (typez==typey) {
      choiceY->SetSelection(0);
      if (typex==3) choiceX->SetSelection(0);
      return;
    }
    // autosetup left/right
    if (typez!=3) AutoLeftRight();
  }


  void SetorientationDialog::AutoLeftRight(){
    int x = this->GetChoiceX();
    int y = this->GetChoiceY();
    int z = this->GetChoiceZ();
    int typex,typey,typez;
    typex = (x+1)/2;  // sup/inf (type=1), ant/pos (type=2), left/right (type=3)
    typey = (y+1)/2;  // not selected (type=0)
    typez = (z+1)/2;
    
    // cross product
    int Ax,Ay,Az,Bx,By,Bz,ABx,ABy,ABz;
    if (x==1) { Ax=0; Ay=-1; Az=0; }
    if (x==2) { Ax=0; Ay=+1; Az=0; }
    if (y==1) { Ax=0; Ay=0; Az=-1; }
    if (y==2) { Ax=0; Ay=0; Az=+1; }
    if (z==1) { Ax=-1; Ay=0; Az=0; }
    if (z==2) { Ax=+1; Ay=0; Az=0; }
    
    if (x==3) { Bx=0; By=-1; Bz=0; }
    if (x==4) { Bx=0; By=+1; Bz=0; }
    if (y==3) { Bx=0; By=0; Bz=-1; }
    if (y==4) { Bx=0; By=0; Bz=+1; }
    if (z==3) { Bx=-1; By=0; Bz=0; }
    if (z==4) { Bx=+1; By=0; Bz=0; }
    
    ABx = Ay*Bz-Az*By;
    ABy = Az*Bx-Ax*Bz;
    ABz = Ax*By-Ay*Bx;
    
    if (ABx==-1) choiceZ->SetSelection(5);
    if (ABx==+1) choiceZ->SetSelection(6);
    if (ABy==-1) choiceX->SetSelection(5);
    if (ABy==+1) choiceX->SetSelection(6);
    if (ABz==-1) choiceY->SetSelection(5);
    if (ABz==+1) choiceY->SetSelection(6);
  }


  int SetorientationDialog::GetChoiceX(){
    return choiceX->GetSelection();
  }

  int SetorientationDialog::GetChoiceY(){
    return choiceY->GetSelection();
  }

  int SetorientationDialog::GetChoiceZ(){
    return choiceZ->GetSelection();
  }

  void SetorientationDialog::GetOrientationString(char s[4]) { // ex. LAS...
    int x = this->GetChoiceX();
    int y = this->GetChoiceY();
    int z = this->GetChoiceZ();
    if (x==1) s[1]='I'; // this is all backwards (its correct)
    if (x==2) s[1]='S';
    if (x==3) s[1]='P';
    if (x==4) s[1]='A';
    if (x==5) s[1]='R';
    if (x==6) s[1]='L';

    if (y==1) s[2]='I';
    if (y==2) s[2]='S';
    if (y==3) s[2]='P';
    if (y==4) s[2]='A';
    if (y==5) s[2]='R';
    if (y==6) s[2]='L';

    if (z==1) s[0]='I';
    if (z==2) s[0]='S';
    if (z==3) s[0]='P';
    if (z==4) s[0]='A';
    if (z==5) s[0]='R';
    if (z==6) s[0]='L';
    s[3]='\0';
  }


} //end Setorientation namespace




