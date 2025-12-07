
#include "manualopt.h"

#include "../xpm/paint.xpm"


namespace Manual{

  ManualOpt :: ManualOpt(wxWindow *parent,
			 ModuleManual *mod)
    : BasePanel(parent){
    wxBitmap *bm[1];

    xhandler = new PaintHandler('x', mod);
    yhandler = new PaintHandler('y', mod);
    zhandler = new PaintHandler('z', mod);

    this->mod = mod;
    bm[0] = new wxBitmap(paint_xpm);

    id_res = APP->idManager->AllocID();
    id_but = APP->idManager->AllocID();
    id_bp  = APP->idManager->AllocID();
    id_fin = APP->idManager->AllocID();
    but = new BitmapRadioButton(this, id_but, bm, 1);
    res = new wxButton(this, id_res, _T("Reset"), wxDefaultPosition, 
		       wxDefaultSize, 0,
		       wxDefaultValidator, _T("butReset"));
    finish = new wxButton(this, id_fin, _T("Finish"),
			  wxDefaultPosition,
			  wxDefaultSize, 0,
			  wxDefaultValidator, _T("butFin"));

    bPicker = new BrushPicker(this, id_bp, true);
    wxStaticText *tbp = new wxStaticText(this, -1, _T("Brush"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText1"));

    wxBoxSizer *hbs2 = new wxBoxSizer(wxHORIZONTAL);
    hbs2->Add(tbp,     0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);
    hbs2->Add(bPicker, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);

    wxStaticBoxSizer *vsizer = new wxStaticBoxSizer(wxVERTICAL, this, 
						    _T("Paint Options"));

    vsizer->AddSpacer(10);
    vsizer->Add(but,    0, wxALIGN_CENTER);
    vsizer->Add(hbs2,   0, wxALIGN_CENTER);
    vsizer->Add(res,    0, wxALIGN_CENTER);
    vsizer->Add(finish, 0, wxALIGN_CENTER);
  
    //sizer->AddSpacer(10);
    sizer->Add(vsizer, 0, wxEXPAND);
    sizer->SetSizeHints(this);
    sizer->Layout();

    Connect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(ManualOpt::OnChangeMode),
	     NULL, NULL );
    Connect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(ManualOpt::OnReset),
	     NULL, NULL );
    Connect( id_fin, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(ManualOpt::OnFinish),
	     NULL, NULL );

  }


  ManualOpt::~ManualOpt(){ 
    Disconnect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(ManualOpt::OnChangeMode),
		NULL, NULL );
    Disconnect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(ManualOpt::OnReset),
		NULL, NULL );
    Disconnect( id_fin, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(ManualOpt::OnFinish),
		NULL, NULL );
    APP->idManager->FreeID(id_res);
    APP->idManager->FreeID(id_but);
    APP->idManager->FreeID(id_bp);
    APP->idManager->FreeID(id_fin);
    APP->SetDefaultInteractionHandler();

    delete xhandler;
    delete yhandler;
    delete zhandler;
  }


  void ManualOpt::OnChangeMode(wxCommandEvent& event){
    ManualOpt::ModeType mode = GetOperationMode();
    
    if(mode==ManualOpt::PAINT){
      APP->Set2DViewInteractionHandler(xhandler, 'x');
      APP->Set2DViewInteractionHandler(yhandler, 'y');
      APP->Set2DViewInteractionHandler(zhandler, 'z');
    }
    else{
      APP->SetDefaultInteractionHandler();
    }
    sizer->Layout();
  }


  ManualOpt::ModeType ManualOpt::GetOperationMode(){
    int sel = but->GetSelection();

    if(sel==0)      return ManualOpt::PAINT;
    else            return ManualOpt::NAVIGATOR;
  }


  void ManualOpt::OnReset(wxCommandEvent& WXUNUSED(event)){
    mod->Reset();
  }


  void ManualOpt::OnFinish(wxCommandEvent& WXUNUSED(event)){
    but->SetSelection(1); // unclick button
    modManager->StopLastActiveProcessingModule();
  }


  AdjRel *ManualOpt :: GetBrush(){
    return bPicker->GetBrush();
  }

  wxCursor *ManualOpt :: GetBrushCursor(int zoom){
    return bPicker->GetBrushCursor(zoom);
  }

  void ManualOpt :: NextBrush(){
    bPicker->NextBrush();
  }

  void ManualOpt :: PrevBrush(){
    bPicker->PrevBrush();
  }

} //end Manual namespace


