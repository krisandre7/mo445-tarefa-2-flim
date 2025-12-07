#include "combinedialog.h"

#include "../xpm/paint.xpm"
#include "../xpm/nav.xpm"
#include "../xpm/and.xpm"
#include "../xpm/or.xpm"
#include "../xpm/neg.xpm"
#include "../xpm/xor.xpm"

#include "../xpm/trash.xpm"

namespace Combine{

  BEGIN_EVENT_TABLE(CombineDialog, BaseDialog)
  EVT_SIZE    (       CombineDialog::OnSize)
  END_EVENT_TABLE()


  CombineDialog :: CombineDialog( wxWindow *parent, ModuleCombine *mod) : BaseDialog( parent, ( char * ) "Combine Paint" ) {
    wxBitmap *bm[6];
    wxSize size(240, 380);
    wxColour wxcolor;

    this->SetMinSize(size);
    this->SetSize(size);
    this->mod = mod;

    panel = new BasePanel(this);
    panel->Show(true);

    xhandler = new CpaintHandler('x', mod);
    yhandler = new CpaintHandler('y', mod);
    zhandler = new CpaintHandler('z', mod);

    bm[ 0 ] = new wxBitmap( nav_xpm );
    bm[ 1 ] = new wxBitmap( paint_xpm );
    bm[ 2 ] = new wxBitmap( and_xpm );
    bm[ 3 ] = new wxBitmap( or_xpm );
    bm[ 4 ] = new wxBitmap( neg_xpm );
    bm[ 5 ] = new wxBitmap( xor_xpm );

    id_but  = APP->idManager->AllocID();
    id_bp   = APP->idManager->AllocID();
    id_res  = APP->idManager->AllocID();
    id_undo = APP->idManager->AllocID();
    id_add  = APP->idManager->AllocID();
    but = new BitmapRadioButton(panel, id_but, bm, 6);
    but->SetSelection(0);
    size.SetHeight(30);
    size.SetWidth(60);
    res = new wxButton(panel, id_res, _T("Reset"), wxDefaultPosition, size, wxBU_EXACTFIT, wxDefaultValidator, _T("butReset"));
    undo = new wxButton(panel, id_undo, _T("Undo"), wxDefaultPosition, size, wxBU_EXACTFIT, wxDefaultValidator, _T("butUndo"));

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
    tools_sizer = new wxStaticBoxSizer(wxVERTICAL, panel, _T("Tools"));
    wxStaticBoxSizer *actions_sizer;
    actions_sizer = new wxStaticBoxSizer(wxVERTICAL, panel, _T("Actions"));

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
    addobj = new wxButton(this, id_add, _T("Add"), wxDefaultPosition, size, wxBU_EXACTFIT, wxDefaultValidator, _T("butAdd"));
    this->sizer->Prepend(addobj, 0, wxALIGN_RIGHT);

    //----------ObjectPanel-------------
    objPanel = this->CreateObjectPanel();
    this->AddPanel((wxPanel *)objPanel);

    if( mod->nobjs > 0 ) {
      mod->obj_sel = 0;
      ChangeObjSelection( v_panel_bkg[ mod->obj_sel ] );
    }
    else
      mod->obj_sel = -1;
    if( mod->nobjs > 1 ) {
      mod->bkg_sel = 1;
      ChangeBkgSelection( v_panel_bkg[ mod->bkg_sel ] );
    }
    else
      mod->bkg_sel = -1;
    
    //----------------------------------

    Connect( id_but, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(CombineDialog::OnChangeMode),
	     NULL, NULL );
    Connect( id_res, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(CombineDialog::OnReset),
	     NULL, NULL );
    Connect( id_undo, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(CombineDialog::OnUndo),
	     NULL, NULL );
    Connect( id_add, wxEVT_COMMAND_BUTTON_CLICKED,
	     wxCommandEventHandler(CombineDialog::OnAddObj),
	     NULL, NULL );
  }


  CombineDialog::~CombineDialog(){ 
    Disconnect( id_but, wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(CombineDialog::OnChangeMode), NULL, NULL );
    Disconnect( id_res, wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(CombineDialog::OnReset), NULL, NULL );
    Disconnect( id_undo, wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(CombineDialog::OnUndo), NULL, NULL );
    Disconnect( id_add, wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(CombineDialog::OnAddObj), NULL, NULL );
    APP->idManager->FreeID( id_but );
    APP->idManager->FreeID( id_bp );
    APP->idManager->FreeID( id_res );
    APP->idManager->FreeID( id_undo );
    APP->idManager->FreeID( id_add );
    APP->SetDefaultInteractionHandler( );

    delete xhandler;
    delete yhandler;
    delete zhandler;

    free( v_panel_bkg );
    free( v_but_color ); 
    free( v_but_trash );
  }

  wxScrolledWindow *CombineDialog::CreateObjectPanel( ) {
    int i,n,id,w,h,wm=200;
    wxSize size(220, 120);
    wxPanel *bkg;

    wxScrolledWindow *swind = new wxScrolledWindow(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 
						   wxHSCROLL | wxVSCROLL, _T("scrolledWindow"));
    swind->SetMinSize(size);
    wxBitmap *bmtrash = new wxBitmap(trash_xpm);

    n = mod->nobjs;
    if( n > 0 ) {
      v_panel_bkg = (wxPanel **)malloc(sizeof(wxPanel *)*n);
      v_but_color = (AlphaColourButton **)malloc(sizeof(AlphaColourButton *)*n);
      v_but_trash = (wxBitmapButton **)malloc(sizeof(wxBitmapButton *)*n);
    }
    else{
      v_panel_bkg = NULL;
      v_but_color = NULL;
      v_but_trash = NULL;
    }

    swind->SetScrollRate(5, 5);
    swind->Scroll(0, 0);
    swind->SetVirtualSize(200,10+MAX(40*n,40));

    for( i = 0; i < n; i++ ) {
      size.SetHeight(wxDefaultCoord);
      size.SetWidth(200);
      bkg = new ObjBkgPanel(swind, wxPoint(10,10+40*i), wxDefaultSize, this);
      bkg->SetMinSize(size);
      v_panel_bkg[i] = bkg;
      v_but_color[i] = new AlphaColourButton(bkg, wxID_ANY);
      v_but_trash[i] = new wxBitmapButton(bkg, wxID_ANY, *bmtrash, wxDefaultPosition, wxDefaultSize, 
					  wxBU_AUTODRAW, wxDefaultValidator, _T("trash"));
      v_but_color[i]->SetValue( mod->myobj[ i ].color );
      v_but_color[i]->SetAlpha( 255 );

      id = v_but_trash[i]->GetId();
      Connect( id, wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(CombineDialog::OnDelete), NULL, NULL );

      id = v_but_color[i]->GetId();
      Connect( id, wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(CombineDialog::OnChangeColor), NULL, NULL );

      wxString *wxstr = new wxString(mod->myobj[ i ].name, wxConvUTF8);
      ObjLabel *tname = new ObjLabel(bkg, *wxstr, this);
      size.SetHeight(wxDefaultCoord);
      size.SetWidth(120);
      tname->SetMinSize(size);
      wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
      hsizer->Add(v_but_trash[i], 0, wxALIGN_RIGHT|wxEXPAND);
      hsizer->Add(v_but_color[i], 0, wxALIGN_LEFT|wxEXPAND);
      hsizer->AddSpacer(10);
      hsizer->Add(tname, 1, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
      
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

    delete bmtrash;

    return (swind);
  }


  void CombineDialog::DestroyObjectPanel(){
    int i,n,id;

    n = mod->nobjs;
    for(i=0; i<n; i++){
      if(v_but_trash==NULL) break;
      id = v_but_trash[i]->GetId();
      Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(CombineDialog::OnDelete), NULL, NULL );
      if(v_but_color==NULL) break;
      id = v_but_color[i]->GetId();
      Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler(CombineDialog::OnChangeColor), NULL, NULL );
    }
    if(v_panel_bkg!=NULL) free(v_panel_bkg);
    if(v_but_color!=NULL) free(v_but_color); 
    if(v_but_trash!=NULL) free(v_but_trash);
    v_panel_bkg = NULL;
    v_but_color = NULL;
    v_but_trash = NULL;

    if(objPanel!=NULL){
      if(sizer->GetItem(objPanel,false)!=NULL)
	sizer->Detach(objPanel);
      objPanel->Destroy();
      objPanel = NULL;
    }
  }


  void CombineDialog :: OnAddObj(wxCommandEvent& event){
    wxSize size(240, 380);
    Scene *label = APP->Data.label;
    int i,r,n,o;
    char msg[ 500 ];
    char name[ 1024 ];
    int color;

    if( mod->nobjs >= MAX_COBJS ) 
      return;

    n = label->n;
    i = mod->nobjs;
    r = APP->ShowNewObjDialog( &color, name );
    if( r != 0 ) 
      return;
    //---------------------------
    for( o = 0; o < mod->nobjs; o++ ) {
      if( strcmp( name, mod->myobj[ o ].name ) == 0 ) {
	sprintf( msg,"You are already segmenting the '%s'.\nChoose a different object.", name );
	wxString wxmsg_w(msg, wxConvUTF8);
	wxMessageBox(wxmsg_w, _T("Warning"), wxOK | wxICON_EXCLAMATION, APP->Window);
	return;
      }
    }

    //---------------------------
    this->DestroyObjectPanel( );

    mod->myobj[ i ].color = color;
    mod->myobj[ i ].visibility = true;
    strcpy( mod->myobj[ i ].name, name );
    mod->myobj[ i ].mask = BMapNew( n );
    mod->nobjs++;

    //APP->Refresh2DCanvas( );
    //APP->Refresh3DCanvas( true, 1.0 );

    this->SetMinSize(size);
    this->SetSize(size);
    objPanel = this->CreateObjectPanel();
    this->AddPanel((wxPanel *)objPanel);
    this->SetMinSize(size);
    this->SetSize(size);

    // Set obj and bkg objects again.
//     mod->obj_sel = -1;
//     mod->bkg_sel = -1;

    if( mod->nobjs == 1 )
      mod->obj_sel = 0;
    
    if( mod->obj_sel >= 0 ) {
      APP->SetLabelColour( 1, mod->myobj[ mod->obj_sel ].color );
      ChangeObjSelection( v_panel_bkg[ mod->obj_sel ] );
    }
    if( mod->bkg_sel >= 0 ) {
      APP->SetLabelColour( 2, mod->myobj[ mod->bkg_sel ].color );
      ChangeBkgSelection( v_panel_bkg[ mod->bkg_sel ] );
    }
    if( ( mod->bkg_sel >= 0 ) && ( mod->obj_sel >= 0 ) )
      APP->SetLabelColour( 3, mod->myobj[ mod->bkg_sel ].color ^ mod->myobj[ mod->obj_sel ].color );
    
    this->RefreshLabel();

    this->Update();
  }

  void CombineDialog :: OnDelete(wxCommandEvent& event){
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
      sprintf(msg,"Do you really wish to delete the %s?",mod->myobj[ i ].name);

      wxString wxmsg(msg, wxConvUTF8);
      wxMessageDialog *dialog = new wxMessageDialog(this, wxmsg, _T("Delete Confirmation"), 
						    wxYES_NO | wxICON_QUESTION, wxDefaultPosition);
      if(dialog->ShowModal() == wxID_YES){
	this->DestroyObjectPanel();
	mod->DeleteObj( i );

	this->SetMinSize(size);
	this->SetSize(size);
	objPanel = this->CreateObjectPanel();
	this->AddPanel((wxPanel *)objPanel);
	if( mod->obj_sel >= 0 )
	  ChangeObjSelection( v_panel_bkg[ mod->obj_sel ] );
	if( mod->bkg_sel >= 0 )
	  ChangeBkgSelection( v_panel_bkg[ mod->bkg_sel ] );


	//free( &mod->myobj[ i ] );
	//this->Fit();
	this->SetMinSize(size);
	this->SetSize(size);
	this->Update();
      }
    }
  }


  void CombineDialog :: OnChangeColor(wxCommandEvent& event){
    int i,n,id;

    id = event.GetId();
    n = mod->nobjs;
    for( i = 0; i < n; i++ ) {
      if( id == v_but_color[ i ]->GetId( ) )
	break;
    }
    if( i < n ){
      mod->myobj[ i ].color = v_but_color[ i ]->GetValue( );
      mod->myobj[ i ].visibility = true;
      if( i == mod->obj_sel ) {
	APP->SetLabelColour( 1, mod->myobj[ i ].color );
	if( mod->bkg_sel >= 0 )
	  APP->SetLabelColour( 3, mod->myobj[ i ].color ^ mod->myobj[ mod->bkg_sel ].color );
      }
      if( i == mod->bkg_sel ) {
	APP->SetLabelColour( 2, mod->myobj[ i ].color );
	if( mod->obj_sel >= 0 )
	  APP->SetLabelColour( 3, mod->myobj[ i ].color ^ mod->myobj[ mod->obj_sel ].color );
      }
      
      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true,1.0);
    }
  }


  void CombineDialog::OnChangeMode(wxCommandEvent& event){
    CombineDialog::ModeType mode = GetOperationMode();
    
    if( mode != CombineDialog::NAVIGATOR ) {
      APP->Set2DViewInteractionHandler(xhandler, 'x');
      APP->Set2DViewInteractionHandler(yhandler, 'y');
      APP->Set2DViewInteractionHandler(zhandler, 'z');
    }
    else{
      APP->SetDefaultInteractionHandler();
    }
    panel->sizer->Layout();
  }


  CombineDialog::ModeType CombineDialog::GetOperationMode(){
    int sel = but->GetSelection();

    xhandler->mode = sel - 1;
    yhandler->mode = sel - 1;
    zhandler->mode = sel - 1;

    if(sel==0)      return CombineDialog::NAVIGATOR;
    else if(sel==1) return CombineDialog::PAINT;
    else if(sel==2) return CombineDialog::AND;
    else if(sel==3) return CombineDialog::OR;
    else if(sel==4) return CombineDialog::NEG;
    else if(sel==5) return CombineDialog::XOR;
    else            return CombineDialog::NAVIGATOR;
  }


  void CombineDialog::OnReset(wxCommandEvent& WXUNUSED(event)){
    mod->Reset();
  }

  void CombineDialog::OnUndo(wxCommandEvent& WXUNUSED(event)){
    //mod->Undo();
  }


  AdjRel *CombineDialog :: GetBrush(){
    return bPicker->GetBrush();
  }

  wxCursor *CombineDialog :: GetBrushCursor(int zoom){
    return bPicker->GetBrushCursor(zoom);
  }

  void CombineDialog :: NextBrush(){
    bPicker->NextBrush();
  }

  void CombineDialog :: PrevBrush(){
    bPicker->PrevBrush();
  }


  void CombineDialog :: OnCancel(wxCommandEvent& event){
    mod->Stop();
    //BaseDialog::OnCancel(event);
  }

  void CombineDialog :: OnOk(wxCommandEvent& event){
    mod->Finish();
    //BaseDialog::OnOk(event);
  }


  void CombineDialog :: OnSize(wxSizeEvent& event){
    BaseDialog::OnSize(event);
  }

  void CombineDialog :: ChangeObjSelection( wxPanel *objbkg ) {
    wxColour wxcolor;
    int i;

    for( i = 0; i < mod->nobjs; i++ ) {
      if( this->v_panel_bkg[ i ] == objbkg ) {
	mod->obj_sel = i;
	SetColor(&wxcolor, 0xffff00);
	this->v_panel_bkg[ i ]->SetBackgroundColour( wxcolor );
	if( i == mod->bkg_sel )
	  mod->bkg_sel = -1;
      }
      else if( mod->bkg_sel != i ) {
	SetColor(&wxcolor, 0xffffff);
	this->v_panel_bkg[ i ]->SetBackgroundColour( wxcolor );
      }
    }
    RefreshLabel();
  }

  void CombineDialog :: ChangeBkgSelection( wxPanel *bkgbkg ) {
    wxColour wxcolor;
    int i;

    for( i = 0; i < mod->nobjs; i++ ) {
      if( this->v_panel_bkg[ i ] == bkgbkg ) {
	mod->bkg_sel = i;
	SetColor( &wxcolor, 0xff00ff );
	this->v_panel_bkg[ i ]->SetBackgroundColour( wxcolor );
	if( i == mod->obj_sel )
	  mod->obj_sel = -1;
      }
      else if( mod->obj_sel != i ) {
	SetColor(&wxcolor, 0xffffff);
	this->v_panel_bkg[ i ]->SetBackgroundColour( wxcolor );
      }
    }
    RefreshLabel();
  }

  void CombineDialog :: RefreshLabel( ) {
    Scene *label;
    int p;

    label = APP->Data.label;
    if( mod->obj_sel >= 0 )
      APP->SetLabelColour( 1, mod->myobj[ mod->obj_sel ].color );
    if( mod->bkg_sel >= 0 )
      APP->SetLabelColour( 2, mod->myobj[ mod->bkg_sel ].color );
    if( ( mod->obj_sel >= 0 ) && ( mod->bkg_sel >= 0 ) )
      APP->SetLabelColour( 3, mod->myobj[ mod->obj_sel ].color ^ mod->myobj[ mod->bkg_sel ].color );
    
    if( ( mod->obj_sel >= 0 ) && ( mod->bkg_sel >= 0 ) ) {
      for( p = 0; p < label->n; p++ )
	label->data[ p ] = ( _fast_BMapGet( mod->myobj[ mod->obj_sel ].mask, p ) ) + 
	                   ( _fast_BMapGet( mod->myobj[ mod->bkg_sel ].mask, p ) * 2 );
    }
    else if( mod->obj_sel >= 0 ) {
      for( p = 0; p < label->n; p++ )
	label->data[ p ] = _fast_BMapGet( mod->myobj[ mod->obj_sel ].mask, p );
    }
    else if( mod->bkg_sel >= 0 ) {
      for( p = 0; p < label->n; p++ )
	label->data[ p ] = _fast_BMapGet( mod->myobj[ mod->bkg_sel ].mask, p ) * 2;
    }
    else {
      for( p = 0; p < label->n; p++ )
	label->data[ p ] = 0;
    }
    
    APP->Refresh2DCanvas( );
    APP->Refresh3DCanvas( true, 1.0 );
  }

  //----------------------------------------------------

  BEGIN_EVENT_TABLE(ObjBkgPanel, wxPanel)
  EVT_MOUSE_EVENTS(ObjBkgPanel::OnMouseEvent)
  END_EVENT_TABLE()

  ObjBkgPanel :: ObjBkgPanel(wxWindow *parent,   wxPoint& pos,   wxSize& size, CombineDialog  *dialog)
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
	dialog->ChangeBkgSelection((wxPanel *)this);
      }
      break;
    }
  }

  //----------------------------------------------------

  BEGIN_EVENT_TABLE(ObjLabel, wxStaticText)
  EVT_MOUSE_EVENTS(ObjLabel::OnMouseEvent)
  END_EVENT_TABLE()

  ObjLabel :: ObjLabel(wxWindow *parent,   wxString& label, CombineDialog  *dialog)
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
	dialog->ChangeBkgSelection((wxPanel *)this->GetParent());
      }
      break;
    }

  }


} //end Combine namespace


