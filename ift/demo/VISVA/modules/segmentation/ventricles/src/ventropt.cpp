
#include "ventropt.h"

#include "../../../xpm/add.xpm"
#include "../../../xpm/del.xpm"

namespace Ventr{

  VentrOpt :: VentrOpt(wxWindow *parent,
		       ModuleVentr *mod)
    : BasePanel(parent){
    wxBitmap *bm[2];

    this->mod = mod;
    bm[0] = new wxBitmap(add_xpm);
    bm[1] = new wxBitmap(del_xpm);

    id_res = APP->idManager->AllocID();
    id_but = APP->idManager->AllocID();
    id_bp  = APP->idManager->AllocID();
    id_fin = APP->idManager->AllocID();
    but = new BitmapRadioButton(this, id_but, bm, 2);
    res = new wxButton(this, id_res, _T("Reset"), wxDefaultPosition, 
		       wxDefaultSize, 0,
		       wxDefaultValidator, _T("butReset"));
    finish = new wxButton(this, id_fin, _T("Finish"),
			  wxDefaultPosition,
			  wxDefaultSize, 0,
			  wxDefaultValidator, _T("butFin"));

    bPicker = new BrushPicker(this, id_bp, true);
    wxStaticText *tbp = new wxStaticText(this, -1, _T("Brush"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText1"));

    xhandler = new AddHandler(mod, 'x', bPicker);
    yhandler = new AddHandler(mod, 'y', bPicker);
    zhandler = new AddHandler(mod, 'z', bPicker);
    dhandler = new DelHandler(mod);

    wxBoxSizer *hbs2 = new wxBoxSizer(wxHORIZONTAL);
    hbs2->Add(tbp,     0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);
    hbs2->Add(bPicker, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);

    wxStaticBoxSizer *vsizer = new wxStaticBoxSizer(wxVERTICAL, this, 
						    _T("Options"));

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
	     wxCommandEventHandler(VentrOpt::OnChangeMode),
	     NULL, NULL );
    Connect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(VentrOpt::OnReset),
	     NULL, NULL );
    Connect( id_fin, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(VentrOpt::OnFinish),
	     NULL, NULL );
  }


  VentrOpt::~VentrOpt(){ 
    Disconnect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(VentrOpt::OnChangeMode),
		NULL, NULL );
    Disconnect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(VentrOpt::OnReset),
		NULL, NULL );
    Disconnect( id_fin, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(VentrOpt::OnFinish),
		NULL, NULL );
    APP->idManager->FreeID(id_res);
    APP->idManager->FreeID(id_but);
    APP->idManager->FreeID(id_bp);
    APP->idManager->FreeID(id_fin);

    APP->Set2DViewInteractionHandler(NULL, 'x');
    APP->Set2DViewInteractionHandler(NULL, 'y');
    APP->Set2DViewInteractionHandler(NULL, 'z');

    delete xhandler;
    delete yhandler;
    delete zhandler;
    delete dhandler;
  }


  void VentrOpt::OnChangeMode(wxCommandEvent& event){
    VentrOpt::ModeType mode = GetOperationMode();

    if(mode==VentrOpt::ADDMARKER){
      APP->Set2DViewInteractionHandler(xhandler, 'x');
      APP->Set2DViewInteractionHandler(yhandler, 'y');
      APP->Set2DViewInteractionHandler(zhandler, 'z');
    }
    else if(mode==VentrOpt::DELMARKER){
      APP->Set2DViewInteractionHandler(dhandler, 'x');
      APP->Set2DViewInteractionHandler(dhandler, 'y');
      APP->Set2DViewInteractionHandler(dhandler, 'z');
    }
    else{
      APP->Set2DViewInteractionHandler(NULL, 'x');
      APP->Set2DViewInteractionHandler(NULL, 'y');
      APP->Set2DViewInteractionHandler(NULL, 'z');
    }
    sizer->Layout();
  }


  VentrOpt::ModeType VentrOpt::GetOperationMode(){
    int sel = but->GetSelection();

    if(sel==0)      return VentrOpt::ADDMARKER;
    else if(sel==1) return VentrOpt::DELMARKER;
    else            return VentrOpt::NAVIGATOR;
  }


  void VentrOpt::OnReset(wxCommandEvent& WXUNUSED(event)){
    mod->Reset();
  }


  void VentrOpt::OnFinish(wxCommandEvent& WXUNUSED(event)){
    mod->Finish();
  }


  AdjRel *VentrOpt :: GetBrush(){
    return bPicker->GetBrush();
  }

} //end Ventr namespace

