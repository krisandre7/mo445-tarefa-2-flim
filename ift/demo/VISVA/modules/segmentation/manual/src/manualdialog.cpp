
#include "manualdialog.h"

#include "../xpm/paint.xpm"
#include "../xpm/nav.xpm"

#include "../xpm/eye.xpm"
#include "../xpm/noteye.xpm"
#include "../xpm/trash.xpm"

namespace Manual{

  BEGIN_EVENT_TABLE(ManualDialog, BaseDialog)
    EVT_SIZE    (       ManualDialog::OnSize)
  END_EVENT_TABLE()


  ManualDialog :: ManualDialog(wxWindow *parent,
			       ModuleManual *mod)
  : BaseDialog(parent, (char *)"Manual Paint"){
    wxBitmap *bm[2];
    wxSize size(240, 380);
    wxColour wxcolor;

    this->SetMinSize(size);
    this->SetSize(size);
    this->mod = mod;

    panel = new BasePanel(this);
    panel->Show(true);

    xhandler = new PaintHandler('x', mod);
    yhandler = new PaintHandler('y', mod);
    zhandler = new PaintHandler('z', mod);

    bm[0] = new wxBitmap(nav_xpm);
    bm[1] = new wxBitmap(paint_xpm);

    id_but  = APP->idManager->AllocID();
    id_bp   = APP->idManager->AllocID();
    id_res  = APP->idManager->AllocID();
    id_undo = APP->idManager->AllocID();
    id_add  = APP->idManager->AllocID();
    but = new BitmapRadioButton(panel, id_but, bm, 2);
    but->SetSelection(0);

    size.SetHeight(30);
    size.SetWidth(60);
    res = new wxButton(panel, id_res, _T("Reset"), wxDefaultPosition, 
		       size, wxBU_EXACTFIT,
		       wxDefaultValidator, _T("butReset"));
    undo = new wxButton(panel, id_undo, _T("Undo"), wxDefaultPosition,
			size, wxBU_EXACTFIT,
			wxDefaultValidator, _T("butUndo"));

    SetColor(&wxcolor, 0xff0000);
    res->SetBackgroundColour(wxcolor);
    SetColor(&wxcolor, 0xffffff);
    res->SetForegroundColour(wxcolor);
    SetColor(&wxcolor, 0xffff00);
    undo->SetBackgroundColour(wxcolor);

    bPicker = new BrushPicker(panel, id_bp, true);
    wxStaticText *tbp = new wxStaticText(panel, -1, _T("Brush"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText1"));

    wxBoxSizer *hbs2 = new wxBoxSizer(wxHORIZONTAL);
    hbs2->Add(tbp,     0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);
    hbs2->Add(bPicker, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);

    wxBoxSizer *hbs3 = new wxBoxSizer(wxHORIZONTAL);
    hbs3->Add(res,  0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);
    hbs3->Add(undo, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);

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
  
    panel->sizer->Add(tools_sizer,   0, wxEXPAND);
    panel->sizer->Add(actions_sizer, 0, wxEXPAND);
    panel->sizer->AddSpacer(20);

    panel->sizer->SetSizeHints(panel);
    panel->sizer->Layout();

    this->AddPanel(panel, 0);
    //----------------------------------
    size.SetHeight(30);
    size.SetWidth(60);
    addobj = new wxButton(this, id_add, _T("Add"), wxDefaultPosition,
			  size, wxBU_EXACTFIT,
			  wxDefaultValidator, _T("butAdd"));
    this->sizer->Prepend(addobj, 0, wxALIGN_RIGHT);

    //----------ObjectPanel-------------
    objPanel = this->CreateObjectPanel();
    this->AddPanel((wxPanel *)objPanel);

    ChangeObjSelection(v_panel_bkg[mod->obj_sel]);
    //----------------------------------

    Connect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(ManualDialog::OnChangeMode),
	     NULL, NULL );
    Connect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(ManualDialog::OnReset),
	     NULL, NULL );
    Connect( id_undo, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(ManualDialog::OnUndo),
	     NULL, NULL );
    Connect( id_add, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(ManualDialog::OnAddObj),
	     NULL, NULL );
  }


  ManualDialog::~ManualDialog(){ 
    Disconnect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(ManualDialog::OnChangeMode),
		NULL, NULL );
    Disconnect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(ManualDialog::OnReset),
		NULL, NULL );
    Disconnect( id_undo, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(ManualDialog::OnUndo),
		NULL, NULL );
    Disconnect( id_add, wxEVT_COMMAND_BUTTON_CLICKED,
		wxCommandEventHandler(ManualDialog::OnAddObj),
		NULL, NULL );
    APP->idManager->FreeID(id_but);
    APP->idManager->FreeID(id_bp);
    APP->idManager->FreeID(id_res);
    APP->idManager->FreeID(id_undo);
    APP->idManager->FreeID(id_add);
    APP->SetDefaultInteractionHandler();

    delete xhandler;
    delete yhandler;
    delete zhandler;

    free(v_panel_bkg);
    free(v_but_color); 
    free(v_but_eye);
    free(v_but_trash);
  }

  wxScrolledWindow *ManualDialog::CreateObjectPanel(){
    int i,n,id,w,h,wm=200;
    wxSize size(220, 120);
    wxPanel *bkg;

    wxScrolledWindow *swind = new wxScrolledWindow(this, wxID_ANY,
						   wxDefaultPosition, 
						   wxDefaultSize, 
						   wxHSCROLL | wxVSCROLL, 
						   _T("scrolledWindow"));
    swind->SetMinSize(size);
    wxBitmap *bmeye   = new wxBitmap(eye_xpm);
    wxBitmap *bmneye  = new wxBitmap(noteye_xpm);
    wxBitmap *bmtrash = new wxBitmap(trash_xpm);

    n = mod->nobjs;
    if(n>0){
      v_panel_bkg = (wxPanel **)malloc(sizeof(wxPanel *)*n);
      v_but_color = (AlphaColourButton **)malloc(sizeof(AlphaColourButton *)*n);
      v_but_eye = (BitmapToggleButton **)malloc(sizeof(BitmapToggleButton *)*n);
      v_but_trash = (wxBitmapButton **)malloc(sizeof(wxBitmapButton *)*n);
    }
    else{
      v_panel_bkg = NULL;
      v_but_color = NULL;
      v_but_eye   = NULL;
      v_but_trash = NULL;
    }

    swind->SetScrollRate(5, 5);
    swind->Scroll(0, 0);
    swind->SetVirtualSize(200,10+MAX(40*n,40));

    for(i=0; i<n; i++){
      size.SetHeight(wxDefaultCoord);
      size.SetWidth(200);
      bkg = new ObjBkgPanel(swind, 
			    wxPoint(10,10+40*i),
			    wxDefaultSize, 
			    this);
      //bkg->SetSize(size);
      bkg->SetMinSize(size);
      v_panel_bkg[i] = bkg;
      v_but_color[i] = new AlphaColourButton(bkg, wxID_ANY);
      v_but_eye[i] = new BitmapToggleButton(bkg, wxID_ANY,
					    bmneye, bmeye);
      v_but_trash[i] = new wxBitmapButton(bkg, wxID_ANY, *bmtrash, 
					  wxDefaultPosition, wxDefaultSize, 
					  wxBU_AUTODRAW, wxDefaultValidator, 
					  _T("trash"));
      v_but_color[i]->SetValue(mod->obj_color[i]);
      v_but_color[i]->SetAlpha(255);
      v_but_eye[i]->SetValue(mod->obj_visibility[i]);

      id = v_but_trash[i]->GetId();
      Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	       wxCommandEventHandler(ManualDialog::OnDelete),
	       NULL, NULL );

      id = v_but_color[i]->GetId();
      Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	       wxCommandEventHandler(ManualDialog::OnChangeColor),
	       NULL, NULL );

      id = v_but_eye[i]->GetId();
      Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	       wxCommandEventHandler(ManualDialog::OnChangeVisibility),
	       NULL, NULL );

      wxString *wxstr = new wxString(mod->obj_name[i], wxConvUTF8);
      ObjLabel *tname = new ObjLabel(bkg, *wxstr, this);
      size.SetHeight(wxDefaultCoord);
      size.SetWidth(120);
      //tname->SetSize(size);
      tname->SetMinSize(size);
      wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
      hsizer->Add(v_but_eye[i],   0, wxALIGN_LEFT|wxEXPAND);
      hsizer->Add(v_but_trash[i], 0, wxALIGN_RIGHT|wxEXPAND);
      hsizer->Add(v_but_color[i], 0, wxALIGN_LEFT|wxEXPAND);
      hsizer->AddSpacer(10);
      hsizer->Add(tname,          1, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
      
      bkg->SetSizer(hsizer, true);
      hsizer->SetSizeHints(bkg);
      hsizer->Layout();

      bkg->GetSize(&w, &h);
      if(w>wm) wm = w;
    }
    for(i=0; i<n; i++){
      size.SetHeight(wxDefaultCoord);
      size.SetWidth(wm);
      (v_panel_bkg[i])->SetSize(size);
    }
    swind->SetVirtualSize(wm,10+MAX(40*n,40));

    wxPaintDC paintDC(swind);
    swind->DoPrepareDC(paintDC);

    delete bmeye;
    delete bmneye;
    delete bmtrash;

    return (swind);
  }


  void ManualDialog::DestroyObjectPanel(){
    int i,n,id;

    n = mod->nobjs;
    for(i=0; i<n; i++){
      if(v_but_trash==NULL) break;
      id = v_but_trash[i]->GetId();
      Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
		  wxCommandEventHandler(ManualDialog::OnDelete),
		  NULL, NULL );
      if(v_but_color==NULL) break;
      id = v_but_color[i]->GetId();
      Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
		  wxCommandEventHandler(ManualDialog::OnChangeColor),
		  NULL, NULL );
      if(v_but_eye==NULL) break;
      id = v_but_eye[i]->GetId();
      Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
		  wxCommandEventHandler(ManualDialog::OnChangeVisibility),
		  NULL, NULL );
    }
    if(v_panel_bkg!=NULL) free(v_panel_bkg);
    if(v_but_color!=NULL) free(v_but_color); 
    if(v_but_eye!=NULL)   free(v_but_eye);
    if(v_but_trash!=NULL) free(v_but_trash);
    v_panel_bkg = NULL;
    v_but_color = NULL;
    v_but_eye   = NULL;
    v_but_trash = NULL;

    if(objPanel!=NULL){
      if(sizer->GetItem(objPanel,false)!=NULL)
	sizer->Detach(objPanel);
      objPanel->Destroy();
      objPanel = NULL;
    }
  }


  void ManualDialog :: OnAddObj(wxCommandEvent& event){
    wxSize size(240, 380);
    SegmObject *obj=NULL;
    Scene *label = APP->Data.label;
    int i,r,p,n,o;
    char title[100];
    char msg[500];
    bool overlap=false;

    if(mod->nobjs>=MAX_OBJS) return;

    n = label->n;
    i = mod->nobjs;
    mod->obj_visibility[i] = true;
    r = APP->ShowNewObjDialog(&mod->obj_color[i],
			      mod->obj_name[i]);
    if (r!=0) return;
    //---------------------------
    for(o=0; o<mod->nobjs; o++){
      if(strcmp(mod->obj_name[i], mod->obj_name[o])==0){
	sprintf(msg,"You are already segmenting the '%s'."\
		    "\nChoose a different object.",
		mod->obj_name[i]);
	wxString wxmsg_w(msg, wxConvUTF8);
	wxMessageBox(wxmsg_w, _T("Warning"),
		     wxOK | wxICON_EXCLAMATION, APP->Window);
	return;
      }
    }

    obj = APP->SearchObjByName(mod->obj_name[i]);
    if(obj!=NULL){
      strcpy(title,(char *)"Resume segmentation?");
      sprintf(msg,"Object '%s' already exists."\
                  "\nResume previous segmentation?"\
                  "\n(Click 'Yes' to resume, or 'No' to start from scratch).",
	      mod->obj_name[i]);
      wxString wxtitle(title, wxConvUTF8);
      wxString wxmsg(msg, wxConvUTF8);
      
      wxMessageDialog dialog(APP->Window,
			     wxmsg, wxtitle,
			     wxYES_NO | wxICON_QUESTION, 
			     wxDefaultPosition); 
      if(dialog.ShowModal() == wxID_YES){

	for(p=0; p<n; p++){
	  if(_fast_BMapGet(obj->mask, p))
	    if(label->data[p]>0)
	      overlap = true;
	}

	if(overlap){
	  strcpy(title,(char *)"Overwrite segmentation?");
	  sprintf(msg,"The '%s' will overwrite some parts of "\
                      "your current segmentation."\
                      "\nContinue anyway?"\
                      "\n(Click 'Yes' to overwrite, or 'No' to abort).",
		  mod->obj_name[i]);
	  wxString wxtitle_o(title, wxConvUTF8);
	  wxString wxmsg_o(msg, wxConvUTF8);
	  
	  wxMessageDialog dialog_o(APP->Window,
				   wxmsg_o, wxtitle_o,
				   wxYES_NO | wxICON_QUESTION, 
				   wxDefaultPosition);
	  if(dialog_o.ShowModal() != wxID_YES)
	    return;
	}

	for(p=0; p<n; p++){
	  if(_fast_BMapGet(obj->mask, p))
	    label->data[p] = i+1;
	}
      }
    }
    //---------------------------
    this->DestroyObjectPanel();

    APP->SetLabelColour(i+1, mod->obj_color[i]);
    mod->nobjs++;
    mod->obj_sel = i;
    if(mod->obj_sel<0) mod->obj_sel = 0;

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);

    this->SetMinSize(size);
    this->SetSize(size);
    objPanel = this->CreateObjectPanel();
    this->AddPanel((wxPanel *)objPanel);
    ChangeObjSelection(v_panel_bkg[mod->nobjs-1]); // last one
    //this->Fit();
    this->SetMinSize(size);
    this->SetSize(size);
    this->Update();
  }


  void ManualDialog :: OnDelete(wxCommandEvent& event){
    wxSize size(240, 380);
    char msg[1024];
    int i,n,id;

    id = event.GetId();
    n = mod->nobjs;
    for(i=0; i<n; i++){
      if(id == v_but_trash[i]->GetId())
	break;
    }
    if(i<n){
      sprintf(msg,"Do you really wish to delete the %s?",mod->obj_name[i]);

      wxString wxmsg(msg, wxConvUTF8);
      wxMessageDialog *dialog = new wxMessageDialog(this, wxmsg, 
						    _T("Delete Confirmation"), 
						    wxYES_NO | wxICON_QUESTION, 
						    wxDefaultPosition);
      if(dialog->ShowModal() == wxID_YES){
	this->DestroyObjectPanel();
	mod->DeleteObj(i);

	this->SetMinSize(size);
	this->SetSize(size);
	objPanel = this->CreateObjectPanel();
	this->AddPanel((wxPanel *)objPanel);
	if(mod->obj_sel>=0 && v_panel_bkg!=NULL)
	  ChangeObjSelection(v_panel_bkg[mod->obj_sel]);
	//this->Fit();
	this->SetMinSize(size);
	this->SetSize(size);
	this->Update();
      }
    }
  }


  void ManualDialog :: OnChangeColor(wxCommandEvent& event){
    int i,n,id;

    id = event.GetId();
    n = mod->nobjs;
    for(i=0; i<n; i++){
      if(id == v_but_color[i]->GetId())
	break;
    }
    if(i<n){
      mod->obj_color[i] = v_but_color[i]->GetValue();
      APP->SetLabelColour(i+1, mod->obj_color[i]);

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true,1.0);
    }
  }


  void ManualDialog :: OnChangeVisibility(wxCommandEvent& event){
    int i,n,id;

    id = event.GetId();
    n = mod->nobjs;
    for(i=0; i<n; i++){
      if(id == v_but_eye[i]->GetId())
	break;
    }
    if(i<n){
      mod->obj_visibility[i] = v_but_eye[i]->GetValue();
      
      if(mod->obj_visibility[i])
	APP->SetLabelColour(i+1, mod->obj_color[i]);
      else
	APP->SetLabelColour(i+1, NIL);

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true,1.0);
    }
  }


  void ManualDialog::OnChangeMode(wxCommandEvent& event){
    ManualDialog::ModeType mode = GetOperationMode();
    
    if(mode==ManualDialog::PAINT){
      APP->Set2DViewInteractionHandler(xhandler, 'x');
      APP->Set2DViewInteractionHandler(yhandler, 'y');
      APP->Set2DViewInteractionHandler(zhandler, 'z');
    }
    else{
      APP->SetDefaultInteractionHandler();
    }
    panel->sizer->Layout();
  }


  ManualDialog::ModeType ManualDialog::GetOperationMode(){
    int sel = but->GetSelection();

    if(sel==0)      return ManualDialog::NAVIGATOR;
    else if(sel==1) return ManualDialog::PAINT;
    else            return ManualDialog::NAVIGATOR;
  }


  void ManualDialog::OnReset(wxCommandEvent& WXUNUSED(event)){
    mod->Reset();
  }

  void ManualDialog::OnUndo(wxCommandEvent& WXUNUSED(event)){
    //mod->Undo();
  }


  AdjRel *ManualDialog :: GetBrush(){
    return bPicker->GetBrush();
  }

  wxCursor *ManualDialog :: GetBrushCursor(int zoom){
    return bPicker->GetBrushCursor(zoom);
  }

  void ManualDialog :: NextBrush(){
    bPicker->NextBrush();
  }

  void ManualDialog :: PrevBrush(){
    bPicker->PrevBrush();
  }


  void ManualDialog :: OnCancel(wxCommandEvent& event){
    mod->Stop();
    //BaseDialog::OnCancel(event);
  }

  void ManualDialog :: OnOk(wxCommandEvent& event){
    mod->Finish();
    //BaseDialog::OnOk(event);
  }


  void ManualDialog :: OnSize(wxSizeEvent& event){
    BaseDialog::OnSize(event);
  }

  void ManualDialog :: ChangeObjSelection(wxPanel *objbkg){
    wxPanel *bkg=NULL;
    wxColour wxcolor;
    int n,i;

    n = mod->nobjs;
    for(i=0; i<n; i++){
      bkg = this->v_panel_bkg[i];
      if(bkg==objbkg){
	mod->obj_sel = i;
	SetColor(&wxcolor, 0xffff00);
      }
      else if(i%2==0) SetColor(&wxcolor, 0xffffff);
      else     	      SetColor(&wxcolor, 0xffffdd);
      bkg->SetBackgroundColour(wxcolor);
    }
  }

  //----------------------------------------------------

BEGIN_EVENT_TABLE(ObjBkgPanel, wxPanel)
  EVT_MOUSE_EVENTS(ObjBkgPanel::OnMouseEvent)
END_EVENT_TABLE()

  ObjBkgPanel :: ObjBkgPanel(wxWindow *parent,
			       wxPoint& pos,
			       wxSize& size,
			     ManualDialog  *dialog)
    : wxPanel(parent, wxID_ANY, pos, size, wxTAB_TRAVERSAL, _T("panel_bkg")){

    this->dialog = dialog;
  }

  ObjBkgPanel :: ~ObjBkgPanel(){
  }

  void ObjBkgPanel :: OnMouseEvent(wxMouseEvent& event){
    switch (event.GetButton()) {
    case wxMOUSE_BTN_LEFT : 
      if(event.GetEventType() == wxEVT_LEFT_DOWN){
	dialog->ChangeObjSelection((wxPanel *)this);
      }
      break;
    case wxMOUSE_BTN_RIGHT : 
      if (event.GetEventType() == wxEVT_RIGHT_DOWN){
	dialog->ChangeObjSelection((wxPanel *)this);
      }
      break;
    }
  }

  //----------------------------------------------------

BEGIN_EVENT_TABLE(ObjLabel, wxStaticText)
  EVT_MOUSE_EVENTS(ObjLabel::OnMouseEvent)
END_EVENT_TABLE()

  ObjLabel :: ObjLabel(wxWindow *parent,
		         wxString& label,
		       ManualDialog  *dialog)
    : wxStaticText(parent, wxID_ANY, label, wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("obj_label")){

    this->dialog = dialog;
  }

  ObjLabel :: ~ObjLabel(){
  }

  void ObjLabel :: OnMouseEvent(wxMouseEvent& event){
    switch (event.GetButton()) {
    case wxMOUSE_BTN_LEFT : 
      if(event.GetEventType() == wxEVT_LEFT_DOWN){
	dialog->ChangeObjSelection((wxPanel *)this->GetParent());
      }
      break;
    case wxMOUSE_BTN_RIGHT : 
      if (event.GetEventType() == wxEVT_RIGHT_DOWN){
	dialog->ChangeObjSelection((wxPanel *)this->GetParent());
      }
      break;
    }
  }


} //end Manual namespace


