
#include "interactiveopt.h"


#include "../xpm/del.xpm"
#include "../xpm/add.xpm"

namespace Interactive{

  InteractiveOpt :: InteractiveOpt(wxWindow *parent,
				   ModuleInteractive *mod)
    : BasePanel(parent){
    wxBitmap *bm[2];

    xhandler = new AddHandler('x', mod);
    yhandler = new AddHandler('y', mod);
    zhandler = new AddHandler('z', mod);
    dhandler = new DelHandler(mod);

    this->mod = mod;
    bm[0] = new wxBitmap(add_xpm);
    bm[1] = new wxBitmap(del_xpm);

    id_res   = APP->idManager->AllocID();
    id_but   = APP->idManager->AllocID();
    id_bp    = APP->idManager->AllocID();
    id_run   = APP->idManager->AllocID();
    id_fin   = APP->idManager->AllocID();

    but = new BitmapRadioButton(this, id_but, bm, 2);
    res = new wxButton(this, id_res, _T("Reset"), wxDefaultPosition, 
		       wxDefaultSize, 0,
		       wxDefaultValidator, _T("butReset"));
    run = new wxButton(this, id_run, _T("Run"),
			  wxDefaultPosition,
			  wxDefaultSize, 0,
			  wxDefaultValidator, _T("butRun"));
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
						    _T("Seg. Options"));

    vsizer->AddSpacer(10);
    vsizer->Add(but,    0, wxALIGN_CENTER);
    vsizer->Add(hbs2,   0, wxALIGN_CENTER);
    vsizer->Add(res,    0, wxALIGN_CENTER);
    vsizer->Add(run, 0, wxALIGN_CENTER);
    vsizer->Add(finish, 0, wxALIGN_CENTER);
  
    //sizer->AddSpacer(10);
    sizer->Add(vsizer, 0, wxEXPAND);
    sizer->SetSizeHints(this);
    sizer->Layout();

    Connect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(InteractiveOpt::OnChangeMode),
	     NULL, NULL );
    Connect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(InteractiveOpt::OnReset),
	     NULL, NULL );
    Connect( id_run, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(InteractiveOpt::OnRun),
	     NULL, NULL );
    Connect( id_fin, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(InteractiveOpt::OnFinish),
	     NULL, NULL );
  }


  InteractiveOpt::~InteractiveOpt(){ 
    Disconnect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(InteractiveOpt::OnChangeMode),
		NULL, NULL );
    Disconnect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(InteractiveOpt::OnReset),
		NULL, NULL );
    Disconnect( id_run, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(InteractiveOpt::OnRun),
		NULL, NULL );
    Disconnect( id_fin, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(InteractiveOpt::OnFinish),
		NULL, NULL );
    APP->idManager->FreeID(id_res);
    APP->idManager->FreeID(id_but);
    APP->idManager->FreeID(id_bp);
    APP->idManager->FreeID(id_run);
    APP->idManager->FreeID(id_fin);
    APP->SetDefaultInteractionHandler();

    delete xhandler;
    delete yhandler;
    delete zhandler;
    delete dhandler;
  }


  void InteractiveOpt::OnChangeMode(wxCommandEvent& event){
    InteractiveOpt::ModeType mode = GetOperationMode();
    
    if(mode==InteractiveOpt::ADDMARKER){
      APP->Set2DViewInteractionHandler(xhandler, 'x');
      APP->Set2DViewInteractionHandler(yhandler, 'y');
      APP->Set2DViewInteractionHandler(zhandler, 'z');
    }
    else if(mode==InteractiveOpt::DELMARKER){
      APP->Set2DViewInteractionHandler(dhandler, 'x');
      APP->Set2DViewInteractionHandler(dhandler, 'y');
      APP->Set2DViewInteractionHandler(dhandler, 'z');
    }
    else{
      APP->SetDefaultInteractionHandler();
    }
    sizer->Layout();
  }


  InteractiveOpt::ModeType InteractiveOpt::GetOperationMode(){
    int sel = but->GetSelection();

    if(sel==0)      return InteractiveOpt::ADDMARKER;
    else if(sel==1) return InteractiveOpt::DELMARKER;
    else            return InteractiveOpt::NAVIGATOR;
  }


  void InteractiveOpt::OnReset(wxCommandEvent& WXUNUSED(event)){
    mod->Reset();
  }


  void InteractiveOpt::OnRun(wxCommandEvent& WXUNUSED(event)){
    mod->Run();
  }

  void InteractiveOpt::OnFinish(wxCommandEvent& WXUNUSED(event)){
    modManager->StopLastActiveProcessingModule();
  }


  AdjRel *InteractiveOpt :: GetBrush(){
    return bPicker->GetBrush();
  }

  wxCursor *InteractiveOpt :: GetBrushCursor(int zoom){
    return bPicker->GetBrushCursor(zoom);
  }

  void InteractiveOpt :: NextBrush(){
    bPicker->NextBrush();
  }

  void InteractiveOpt :: PrevBrush(){
    bPicker->PrevBrush();
  }


} //end Interactive namespace


