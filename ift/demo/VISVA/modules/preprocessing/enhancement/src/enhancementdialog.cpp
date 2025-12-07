
#include "enhancementdialog.h"


#include "../xpm/del.xpm"
#include "../xpm/add.xpm"
#include "../xpm/nav.xpm"


namespace Enhancement{

  BEGIN_EVENT_TABLE(EnhancementDialog, BaseDialog)
    EVT_SIZE    (       EnhancementDialog::OnSize)
  END_EVENT_TABLE()


  EnhancementDialog :: EnhancementDialog(wxWindow *parent,
					 ModuleEnhancement *mod)
  : BaseDialog(parent, (char *)"Object Enhancement"){
    wxBitmap *bm[3];
    wxSize size(240, 380);
    wxColour wxcolor;

    this->SetMinSize(size);
    this->SetSize(size);
    this->mod = mod;

    panel = new BasePanel(this);
    panel->Show(true);

    xhandler = new AddSamples('x', mod);
    yhandler = new AddSamples('y', mod);
    zhandler = new AddSamples('z', mod);
    dhandler = new DelSamples(mod);

    bm[0] = new wxBitmap(nav_xpm);
    bm[1] = new wxBitmap(add_xpm);
    bm[2] = new wxBitmap(del_xpm);

    id_but  = APP->idManager->AllocID();
    id_bp   = APP->idManager->AllocID();
    id_res  = APP->idManager->AllocID();
    id_undo = APP->idManager->AllocID();
    id_run  = APP->idManager->AllocID();
    but = new BitmapRadioButton(panel, id_but, bm, 3);
    but->SetSelection(0);

    size.SetHeight(30);
    size.SetWidth(60);
    res = new wxButton(panel, id_res, _T("Reset"), wxDefaultPosition, 
		       size, wxBU_EXACTFIT,
		       wxDefaultValidator, _T("butReset"));
    undo = new wxButton(panel, id_undo, _T("Undo"), wxDefaultPosition,
			size, wxBU_EXACTFIT,
			wxDefaultValidator, _T("butUndo"));
    run = new wxButton(panel, id_run, _T("Run"), wxDefaultPosition,
		       size, wxBU_EXACTFIT,
		       wxDefaultValidator, _T("butRun"));

    SetColor(&wxcolor, 0xff0000);
    res->SetBackgroundColour(wxcolor);
    SetColor(&wxcolor, 0xffffff);
    res->SetForegroundColour(wxcolor);
    SetColor(&wxcolor, 0xffff00);
    undo->SetBackgroundColour(wxcolor);
    SetColor(&wxcolor, 0x00ff00);
    run->SetBackgroundColour(wxcolor);

    bPicker = new BrushPicker(panel, id_bp, true);
    wxStaticText *tbp = new wxStaticText(panel, -1, _T("Brush"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText1"));

    wxBoxSizer *hbs2 = new wxBoxSizer(wxHORIZONTAL);
    hbs2->Add(tbp,     0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);
    hbs2->Add(bPicker, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);

    wxBoxSizer *hbs3 = new wxBoxSizer(wxHORIZONTAL);
    hbs3->Add(res,  0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);
    hbs3->Add(undo, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);
    hbs3->Add(run,  0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);

    wxStaticBoxSizer *tools_sizer;
    tools_sizer = new wxStaticBoxSizer(wxVERTICAL, panel, 
				       _T("Tools"));
    wxStaticBoxSizer *actions_sizer;
    actions_sizer = new wxStaticBoxSizer(wxVERTICAL, panel, 
					 _T("Actions"));

    tools_sizer->AddSpacer(10);
    tools_sizer->Add(but,    0, wxALIGN_CENTER);
    tools_sizer->Add(hbs2,   0, wxALIGN_CENTER);

    actions_sizer->Add(hbs3,   0, wxALIGN_CENTER);
  
    panel->sizer->Add(actions_sizer, 0, wxEXPAND);
    panel->sizer->Add(tools_sizer,   0, wxEXPAND);
    panel->sizer->AddSpacer(20);

    panel->sizer->SetSizeHints(panel);
    panel->sizer->Layout();

    this->AddPanel(panel, 0);
    //----------------------------------

    Connect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(EnhancementDialog::OnChangeMode),
	     NULL, NULL );
    Connect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(EnhancementDialog::OnReset),
	     NULL, NULL );
    Connect( id_undo, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(EnhancementDialog::OnUndo),
	     NULL, NULL );
    Connect( id_run, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(EnhancementDialog::OnRun),
	     NULL, NULL );
  }


  EnhancementDialog::~EnhancementDialog(){ 
    Disconnect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(EnhancementDialog::OnChangeMode),
		NULL, NULL );
    Disconnect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(EnhancementDialog::OnReset),
		NULL, NULL );
    Disconnect( id_undo, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(EnhancementDialog::OnUndo),
		NULL, NULL );
    Disconnect( id_run, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(EnhancementDialog::OnRun),
		NULL, NULL );
    APP->idManager->FreeID(id_but);
    APP->idManager->FreeID(id_bp);
    APP->idManager->FreeID(id_res);
    APP->idManager->FreeID(id_undo);
    APP->idManager->FreeID(id_run);
    APP->SetDefaultInteractionHandler();

    delete xhandler;
    delete yhandler;
    delete zhandler;
    delete dhandler;
  }


  void EnhancementDialog::OnChangeMode(wxCommandEvent& event){
    EnhancementDialog::ModeType mode = GetOperationMode();
    
    if(mode==EnhancementDialog::ADDMARKER){
      APP->Set2DViewInteractionHandler(xhandler, 'x');
      APP->Set2DViewInteractionHandler(yhandler, 'y');
      APP->Set2DViewInteractionHandler(zhandler, 'z');
    }
    else if(mode==EnhancementDialog::DELMARKER){
      APP->Set2DViewInteractionHandler(dhandler, 'x');
      APP->Set2DViewInteractionHandler(dhandler, 'y');
      APP->Set2DViewInteractionHandler(dhandler, 'z');
    }
    else{
      APP->SetDefaultInteractionHandler();
    }
    panel->sizer->Layout();
  }


  EnhancementDialog::ModeType EnhancementDialog::GetOperationMode(){
    int sel = but->GetSelection();

    if(sel==0)      return EnhancementDialog::NAVIGATOR;
    else if(sel==1) return EnhancementDialog::ADDMARKER;
    else if(sel==2) return EnhancementDialog::DELMARKER;
    else            return EnhancementDialog::NAVIGATOR;
  }


  void EnhancementDialog::OnReset(wxCommandEvent& WXUNUSED(event)){
    mod->Reset();
  }


  void EnhancementDialog::OnRun(wxCommandEvent& WXUNUSED(event)){
    mod->Run();
  }

  void EnhancementDialog::OnUndo(wxCommandEvent& WXUNUSED(event)){
    //mod->Undo();
  }


  /*
  void InteractiveDialog::OnFinish(wxCommandEvent& WXUNUSED(event)){
    mod->Finish();
  }
  */

  AdjRel *EnhancementDialog :: GetBrush(){
    return bPicker->GetBrush();
  }

  wxCursor *EnhancementDialog :: GetBrushCursor(int zoom){
    return bPicker->GetBrushCursor(zoom);
  }

  void EnhancementDialog :: NextBrush(){
    bPicker->NextBrush();
  }

  void EnhancementDialog :: PrevBrush(){
    bPicker->PrevBrush();
  }


  void EnhancementDialog :: OnCancel(wxCommandEvent& event){
    mod->Stop();
    //BaseDialog::OnCancel(event);
  }

  void EnhancementDialog :: OnOk(wxCommandEvent& event){
    mod->Finish();
    //BaseDialog::OnOk(event);
  }


  void EnhancementDialog :: OnSize(wxSizeEvent& event){
    BaseDialog::OnSize(event);
  }


} //end Enhancement namespace


