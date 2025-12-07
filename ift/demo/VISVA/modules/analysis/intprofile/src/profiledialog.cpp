
#include "profiledialog.h"
#include "poshandler.h"

namespace IntensityProfile{


  BEGIN_EVENT_TABLE(ProfileDialog, BaseDialog)
    EVT_SIZE    (           ProfileDialog::OnSize)
  END_EVENT_TABLE()


  ProfileDialog::ProfileDialog(wxWindow *parent, 
			       int *line, int n,
			       ModuleIntensityProfile *mod)
  : BaseDialog(parent, (char *)"Profile Dialog"){
    wxSize size(480, 180);
    BasePanel  *panel;
    int i;

    position = 0;
    this->SetMinSize(size);
    this->SetSize(size);
    this->mod = mod;

    this->voxels = AllocIntArray(n);
    memcpy(this->voxels, line, n*sizeof(int));

    this->curve1 = CreateCurve(n);
    for(i=0; i<n; i++){
      curve1->X[i] = i;
      curve1->Y[i] = (APP->Data.orig)->data[line[i]];
    }

    if(SceneImax(APP->Data.arcw)>0){
      this->curve2 = CreateCurve(n);
      for(i=0; i<n; i++){
	curve2->X[i] = i;
	curve2->Y[i] = (APP->Data.arcw)->data[line[i]];
      }
    }
    else curve2 = NULL;

    panel = new BasePanel(this);
    panel->Show(true);
    
    plot = new Canvas(panel);
    panel->sizer->Add(plot, 1, wxEXPAND);    

    this->DrawProfile();
    
    panel->sizer->SetSizeHints(panel);
    panel->sizer->Layout();

    if(curve2!=NULL)
      size.SetHeight(360);
    this->SetMinSize(size);
    this->SetSize(size);
    this->AddPanel(panel);
    this->SetMinSize(size);
    this->SetSize(size);

    handler = new PosHandler(this);
    plot->SetInteractionHandler(handler);
  }
  
  
  ProfileDialog :: ~ProfileDialog(){
    PosHandler *phandler=NULL;
    if(curve1!=NULL) DestroyCurve(&curve1);
    if(curve2!=NULL) DestroyCurve(&curve2);
    free(this->voxels);
    plot->SetInteractionHandler(NULL);
    phandler = (PosHandler *)handler;
    delete phandler;
  }
  
  
  void ProfileDialog::DrawProfile(){
    CImage *cimg=NULL,*cimg1,*cimg2;
    PlotRange rangex;
    PlotLandmark marks[3];
    int w,h;

    marks[0].X = (double)position;
    marks[0].color = 0xff0000;
    strcpy(marks[0].name, "");

    //plot->GetSize(&w, &h);
    plot->GetClientSize(&w, &h);
    rangex.begin = 0;
    rangex.end = curve1->n;

    cimg = CreateCImage(w, h);

    if(curve2!=NULL) h /= 2;
    cimg1 = PlotCurve(curve1, w, h,
		      &rangex, NULL,
		      (char *)"Profile",
		      (char *)"Samples",
		      (char *)"Intensity",
		      0xffff00, marks, 1);
    PasteSubCImage(cimg, cimg1, 0, 0);
    DestroyCImage(&cimg1);

    if(curve2!=NULL){
      cimg2 = PlotCurve(curve2, w, h,
			&rangex, NULL,
			(char *)"",
			(char *)"Samples",
			(char *)"ArcWeight",
			0xff00ff, marks, 1);
      PasteSubCImage(cimg, cimg2, 0, h);
      DestroyCImage(&cimg2);
    }

    plot->DrawCImage(cimg);
    plot->Refresh();
    DestroyCImage(&cimg);
  }


  void ProfileDialog :: OnCancel(wxCommandEvent& event){
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();
    BaseDialog::OnCancel(event);
  }

  void ProfileDialog :: OnOk(wxCommandEvent& event){
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();
    BaseDialog::OnOk(event);
  }


  void ProfileDialog :: OnSize(wxSizeEvent& event){
    BaseDialog::OnSize(event);
    this->DrawProfile();
  }

  void ProfileDialog :: SetPosition(int position){
    int p;
    p = this->voxels[this->position];
    //APP->AddSeed(p, 1, 1);
    (APP->Data.label)->data[p] = 1;
    p = this->voxels[position];
    //APP->AddSeed(p, 2, 1);
    (APP->Data.label)->data[p] = 2;

    this->position = position;
    this->DrawProfile();
    APP->Refresh2DCanvas();
  }

  int ProfileDialog :: GetPosition(){
    return position;
  }

  int  ProfileDialog :: GetProfileLength(){
    return curve1->n;
  }


  void ProfileDialog :: GetPlotSize(int *w, int *h){
    plot->GetClientSize(w, h);
  }


} //end IntensityProfile namespace

