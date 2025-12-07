
#include "thresholddialog.h"

namespace ModThreshold{


  BEGIN_EVENT_TABLE(ThresholdDialog, BaseDialog)
    EVT_SIZE    (           ThresholdDialog::OnSize)
  END_EVENT_TABLE()


  ThrRefreshHandler::ThrRefreshHandler(ThresholdDialog *dialog){
    this->dialog = dialog;    
  }

  ThrRefreshHandler::~ThrRefreshHandler(){
  }

  void ThrRefreshHandler::OnRefresh2D(char axis){
    Scene *label = APP->Data.label;
    Scene *scn = APP->Data.orig;
    int p,i,j,k;
    Voxel vs;
    int lower, higher;

    vs = APP->Get2DViewSlice();
    dialog->GetThreshold(&lower, &higher);

    if(axis == 'x'){
      j = vs.x;
      for(k=0; k < label->zsize; k++){
	for(i=0; i < label->ysize; i++){	  
	  p = VoxelAddress(label, j, i, k);
	  if(scn->data[p]>=lower && scn->data[p]<=higher)
	    label->data[ p ] = 1;
	  else
	    label->data[ p ] = 0;
	}
      }
    }
    else if(axis == 'y'){
      i = vs.y;
      for(k=0; k < label->zsize; k++){
	for(j=0; j < label->xsize; j++){
	  p = VoxelAddress(label, j, i, k);
	  if(scn->data[p]>=lower && scn->data[p]<=higher)
	    label->data[ p ] = 1;
	  else
	    label->data[ p ] = 0;
	}
      }
    }
    else{ // axis=='z'
      k = vs.z;
      for(i=0; i < label->ysize; i++){
	for(j=0; j < label->xsize; j++){
	  p = VoxelAddress(label, j, i, k);
	  if(scn->data[p]>=lower && scn->data[p]<=higher)
	    label->data[ p ] = 1;
	  else
	    label->data[ p ] = 0;
	}
      }
    }
  }


  ThresholdDialog::ThresholdDialog(wxWindow *parent, 
				   wxWindowID id1,
				   wxWindowID id2,
				   wxWindowID id3,
				   int Min, int Max,
				   ModuleThreshold *mod)
     : BaseDialog(parent, (char *)"Threshold Dialog"){

     wxSize size(480, 360);
     BasePanel  *panel;
     
     this->mod   = mod;
     this->hist  = Histogram3(APP->Data.orig);
     this->mhist = MedianFilterHistogram( this->hist );
     this->otsu  = ComputeOtsu3(APP->Data.orig);

     panel = new BasePanel(this);
     panel->Show(true);
    
    plot = new SimpleCanvas(panel);
    
    spinLower1  = new SpinSlider(panel, id1, Min,   Min, Max);
    spinHigher1 = new SpinSlider(panel, id1, Max,   Min, Max);
    spinLower2  = new SpinSlider(panel, id2, Min+1, Min, Max);
    spinHigher2 = new SpinSlider(panel, id2, Max,   Min, Max);

    wxStaticText *tl1 = new wxStaticText(panel, -1, _T("Lower"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE, _T("staticText1"));

    wxStaticText *th1 = new wxStaticText(panel, -1, _T("Higher"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE, _T("staticText2"));

   wxStaticText *tl2 = new wxStaticText(panel, -1, _T("Lower"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE, _T("staticText3"));

    wxStaticText *th2 = new wxStaticText(panel, -1, _T("Higher"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE, _T("staticText4"));

    wxStaticBoxSizer *sbs1 = new wxStaticBoxSizer(wxVERTICAL,
						  panel,
						  _T("Threshold"));
    wxStaticBoxSizer *sbs2 = new wxStaticBoxSizer(wxVERTICAL,
						  panel,
						  _T("Zoom interval"));
    
    panel->sizer->Add(plot, 1, wxEXPAND);    

    wxString *objChoices;
    objChoices = new wxString[ 2 ];
    objChoices[ 0 ] = wxString::FromAscii( "Histogram" );
    objChoices[ 1 ] = wxString::FromAscii( "Median Histogram" );

    histogramTypeTxt = new wxStaticText(panel, wxID_ANY, _T("Select Histogram:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    histogramType = new wxChoice(panel, id3, wxDefaultPosition, wxDefaultSize, 2, (wxString *) objChoices, 0, wxDefaultValidator, _T("choice0"));
    histogramType->SetSelection( 0 );

    wxBoxSizer *hbs0 = new wxBoxSizer(wxHORIZONTAL);
    hbs0->Add(histogramTypeTxt, 1, wxEXPAND);
    hbs0->Add(histogramType, 1, wxEXPAND);
    panel->sizer->Add(hbs0, 0, wxEXPAND);

    wxBoxSizer *hbsl1 = new wxBoxSizer(wxHORIZONTAL);
    hbsl1->Add(tl1, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsl1->Add(spinLower1, 1, wxALIGN_LEFT);
    sbs1->Add(hbsl1, 0, wxEXPAND);

    wxBoxSizer *hbsh1 = new wxBoxSizer(wxHORIZONTAL);
    hbsh1->Add(th1, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsh1->Add(spinHigher1, 1, wxALIGN_LEFT);
    sbs1->Add(hbsh1, 0, wxEXPAND);

    wxBoxSizer *hbsl2 = new wxBoxSizer(wxHORIZONTAL);
    hbsl2->Add(tl2, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsl2->Add(spinLower2, 1, wxALIGN_LEFT);
    sbs2->Add(hbsl2, 0, wxEXPAND);

    wxBoxSizer *hbsh2 = new wxBoxSizer(wxHORIZONTAL);
    hbsh2->Add(th2, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
    hbsh2->Add(spinHigher2, 1, wxALIGN_LEFT);
    sbs2->Add(hbsh2, 0, wxEXPAND);


    wxBoxSizer *hbs = new wxBoxSizer(wxHORIZONTAL);
    hbs->Add(sbs1, 1, wxEXPAND);
    hbs->Add(sbs2, 1, wxEXPAND);
    panel->sizer->Add(hbs, 0, wxEXPAND);

    Connect( id1, wxEVT_SCROLL_THUMBRELEASE,
	     wxScrollEventHandler(ThresholdDialog::OnThresholdChange),
	     NULL, NULL );
    Connect( id1, wxEVT_SCROLL_THUMBTRACK,
	     wxScrollEventHandler(ThresholdDialog::OnThresholdChange),
	     NULL, NULL );

    Connect( id2, wxEVT_SCROLL_THUMBRELEASE,
	     wxScrollEventHandler(ThresholdDialog::OnIntervalChange),
	     NULL, NULL );
    Connect( id2, wxEVT_SCROLL_THUMBTRACK,
	     wxScrollEventHandler(ThresholdDialog::OnIntervalChange),
	     NULL, NULL );

    Connect( id3, wxEVT_COMMAND_CHOICE_SELECTED,
             wxCommandEventHandler(ThresholdDialog::OnChangeHistogram),
             NULL, NULL );

    this->DrawHistogram();
    
    panel->sizer->SetSizeHints(panel);
    panel->sizer->Layout();
    this->SetMinSize(size);
    this->SetSize(size);
    this->AddPanel(panel);
    this->SetMinSize(size);
    this->SetSize(size);

    this->handler = new ThrRefreshHandler(this);
    APP->SetRefresh2DHandler(this->handler);
    APP->Refresh2DCanvas();
    //APP->Refresh3DCanvas(true, 1.0);
  }
  
  
  ThresholdDialog :: ~ThresholdDialog(){
    int id1 = spinLower1->GetId();
    int id2 = spinLower2->GetId();
    int id3 = histogramType->GetId();

    Disconnect( id1, wxEVT_SCROLL_THUMBRELEASE,
		wxScrollEventHandler(ThresholdDialog::OnThresholdChange),
		NULL, NULL );
    Disconnect( id1, wxEVT_SCROLL_THUMBTRACK,
		wxScrollEventHandler(ThresholdDialog::OnThresholdChange),
		NULL, NULL );

    Disconnect( id2, wxEVT_SCROLL_THUMBRELEASE,
		wxScrollEventHandler(ThresholdDialog::OnIntervalChange),
		NULL, NULL );
    Disconnect( id2, wxEVT_SCROLL_THUMBTRACK,
		wxScrollEventHandler(ThresholdDialog::OnIntervalChange),
		NULL, NULL );

    Disconnect( id3, wxEVT_COMMAND_CHOICE_SELECTED,
             wxCommandEventHandler(ThresholdDialog::OnChangeHistogram),
             NULL, NULL );


    DestroyCurve(&hist);
    DestroyCurve(&mhist);

    APP->SetRefresh2DHandler(NULL);
    delete this->handler;
    this->handler = NULL;
  }
  
  
  void ThresholdDialog::GetThreshold(int *lower, int *higher){
    *lower  = spinLower1->GetValue();
    *higher = spinHigher1->GetValue();
  }


  void ThresholdDialog::RefreshObject(){
    Scene *label = APP->Data.label;
    Scene *scn = APP->Data.orig;
    int p,lower,higher;

    this->GetThreshold(&lower, &higher);
    for(p=0; p<scn->n; p++){
      if(scn->data[p]>=lower && scn->data[p]<=higher)
	label->data[p] = 1;
      else
	label->data[p] = 0;
    }
  }


  void ThresholdDialog::DrawHistogram(){
    CImage *cimg=NULL;
    PlotRange rangex;
    PlotLandmark marks[3];
    int w,h;
    int x,i;
    char mark0_y[ 10 ];
    char mark1_y[ 10 ];

    if( strcmp( histogramType->GetStringSelection().ToAscii(), "Histogram" ) == 0 )
      sprintf(mark0_y, "%d", ( int ) this->hist->Y[ spinLower1->GetValue() ]);
    else {
      x = spinLower1->GetValue();
      for( i = 0; ( this->mhist->X[ i ] < x ) && ( i < this->mhist->n - 1 ); i++ );
      sprintf(mark0_y, "%d", ( int ) this->mhist->Y[ i ]);
    }
    marks[0].X = spinLower1->GetValue();
    marks[0].color = 0xff0000;
    strcpy(marks[0].name, mark0_y);

    if( strcmp( histogramType->GetStringSelection().ToAscii(), "Histogram" ) == 0 )
      sprintf(mark1_y, "%d", ( int ) this->hist->Y[ spinHigher1->GetValue() ]);
    else{
      x = spinHigher1->GetValue();
      for( i = 0; ( this->mhist->X[ i ] < x ) && ( i < this->mhist->n - 1 ); i++ );
      sprintf(mark1_y, "%d", ( int ) this->mhist->Y[ i ]);
    }
      
    marks[1].X = spinHigher1->GetValue();
    marks[1].color = 0xff0000;
    strcpy(marks[1].name, mark1_y);

    marks[2].X = (double)this->otsu;
    marks[2].color = 0x00ff00;
    strcpy(marks[2].name, "Otsu");

    plot->GetSize(&w, &h);
    rangex.begin = spinLower2->GetValue();
    rangex.end = spinHigher2->GetValue();
    if( strcmp( histogramType->GetStringSelection().ToAscii(), "Histogram" ) == 0 )
      cimg = PlotCurve(hist, w, h, &rangex, NULL, (char *)"Histogram", (char *)"Intensity", 
		       (char *)"Frequency",0xffff00, marks, 3);
    else
      cimg = PlotCurve(mhist, w, h, &rangex, NULL, (char *)"Histogram", (char *)"Intensity", 
		       (char *)"Frequency",0xffff00, marks, 3);
    plot->DrawCImage(cimg);
    plot->Refresh();
    DestroyCImage(&cimg);
  }


  void ThresholdDialog :: OnThresholdChange(wxScrollEvent& event){
    APP->Refresh2DCanvas();
    //APP->Refresh3DCanvas(true, 1.0);
    this->DrawHistogram();
  }

  void ThresholdDialog :: OnIntervalChange(wxScrollEvent& event){
    this->DrawHistogram();
  }

  void ThresholdDialog :: OnCancel(wxCommandEvent& event){
    //BaseDialog::OnCancel(event);
    mod->Stop();
  }

  void ThresholdDialog :: OnOk(wxCommandEvent& event){
    int p,n;
    Scene *label = APP->Data.label;
    SegmObject *obj;

    this->RefreshObject();

    n = APP->Data.w*APP->Data.h*APP->Data.nframes;

    obj = CreateSegmObject(mod->obj_name,
			   mod->obj_color);

    obj->mask = BMapNew(n);
    BMapFill(obj->mask, 0);
    for(p=0; p<n; p++){
      if(APP->GetLabelColour(label->data[p])!=NIL)
	_fast_BMapSet1(obj->mask, p);
    }
    obj->seed = APP->CopyMarkerList();

    APP->AddCustomObj(obj);

    //APP->SetDefaultInteractionHandler();
    //APP->EnableObjWindow(true);
    //APP->ResetData();
    //APP->DrawSegmObjects();
    //BaseDialog::OnOk(event);
    mod->Stop();
  }

  void ThresholdDialog :: OnSize(wxSizeEvent& event){
    BaseDialog::OnSize(event);
    this->DrawHistogram();
  }

  void ThresholdDialog :: OnChangeHistogram(wxCommandEvent& event){
    this->DrawHistogram();
  }

} //end ModThreshold namespace
