
#include "roidialog.h"

namespace ModROI{


  BEGIN_EVENT_TABLE(ROIDialog, BaseDialog)
    EVT_SIZE    (           ROIDialog::OnSize)
  END_EVENT_TABLE()


  ROIRefreshHandler::ROIRefreshHandler(ROIDialog *dialog){
    this->dialog = dialog;
  }

  ROIRefreshHandler::~ROIRefreshHandler(){
  }


  void ROIRefreshHandler::OnRefresh2D(char axis){
    Scene *label = APP->Data.label;
    int p,i,j,k;
    Voxel v1, v2, vs;

    vs = APP->Get2DViewSlice();
    dialog->GetROI(&v1, &v2);

    if(axis == 'x'){
      j = vs.x;
      for(k=0; k < label->zsize; k++){
	for(i=0; i < label->ysize; i++){	  
	  p = VoxelAddress(label, j, i, k);
	  if( ( ( j >= v1.x ) && ( j <= v2.x ) ) && 
	      ( ( i >= v1.y ) && ( i <= v2.y ) ) &&
	      ( ( k >= v1.z ) && ( k <= v2.z ) ) )
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
	  if( ( ( j >= v1.x ) && ( j <= v2.x ) ) && 
	      ( ( i >= v1.y ) && ( i <= v2.y ) ) &&
	      ( ( k >= v1.z ) && ( k <= v2.z ) ) )
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
	  if( ( ( j >= v1.x ) && ( j <= v2.x ) ) && 
	      ( ( i >= v1.y ) && ( i <= v2.y ) ) &&
	      ( ( k >= v1.z ) && ( k <= v2.z ) ) )
	    label->data[ p ] = 1;
	  else
	    label->data[ p ] = 0;
	}
      }
    }
  }


  ROIDialog::ROIDialog(wxWindow *parent, wxWindowID idx, wxWindowID idy, wxWindowID idz, Scene *scn, ModuleROI *mod)
    : BaseDialog(parent, (char *)"R.O.I. Dialog") {

    wxSizer *sbutton;
    wxBoxSizer *mainsizer;
    wxGridSizer *sizer;
    wxStaticText *tmsg_x1;
    wxStaticText *tmsg_x2;
    wxStaticText *tmsg_y1;
    wxStaticText *tmsg_y2;
    wxStaticText *tmsg_z1;
    wxStaticText *tmsg_z2;

    this->mod = mod;
    mainsizer = new wxBoxSizer(wxVERTICAL);
    sizer = new wxGridSizer(4,0,0);
    sbutton = this->CreateButtonSizer(wxOK|wxCANCEL);

    tmsg_x1 = new wxStaticText(this, idx, _T("Left bound:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    tmsg_x2 = new wxStaticText(this, idx, _T("Right bound:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    tmsg_y1 = new wxStaticText(this, idy, _T("Upper bound:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    tmsg_y2 = new wxStaticText(this, idy, _T("Lower bound:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    tmsg_z1 = new wxStaticText(this, idz, _T("Front bound:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));
    tmsg_z2 = new wxStaticText(this, idz, _T("Rear bound:"), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("staticText"));

    x1 = new wxSpinCtrl( this, idx, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, scn->xsize-1, scn->xsize / 4, _T("LB") );
    x2 = new wxSpinCtrl( this, idx, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, scn->xsize-1, 3 * scn->xsize / 4, _T("RB") );
    y1 = new wxSpinCtrl( this, idy, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, scn->ysize-1, scn->ysize / 4, _T("UB") );
    y2 = new wxSpinCtrl( this, idy, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, scn->ysize-1, 3 * scn->ysize / 4, _T("LB") );
    z1 = new wxSpinCtrl( this, idz, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, scn->zsize-1, scn->zsize / 4, _T("FB") );
    z2 = new wxSpinCtrl( this, idz, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, scn->zsize-1, 3 * scn->zsize / 4, _T("BB") );


    sizer->Add(tmsg_x1, 0, wxEXPAND);
    sizer->Add(x1, 0, wxEXPAND);
    sizer->Add(tmsg_x2, 0, wxEXPAND);
    sizer->Add(x2, 0, wxEXPAND);
    sizer->Add(tmsg_y1, 0, wxEXPAND);
    sizer->Add(y1, 0, wxEXPAND);
    sizer->Add(tmsg_y2, 0, wxEXPAND);
    sizer->Add(y2, 0, wxEXPAND);
    sizer->Add(tmsg_z1, 0, wxEXPAND);
    sizer->Add(z1, 0, wxEXPAND);
    sizer->Add(tmsg_z2, 0, wxEXPAND);
    sizer->Add(z2, 0, wxEXPAND);
    mainsizer->Add(sizer,   0, wxEXPAND);
    //mainsizer->Add(sline,   0, wxEXPAND);
    mainsizer->Add(sbutton, 0, wxEXPAND);

    this->SetSizer(mainsizer, true);
    mainsizer->SetSizeHints(this);
    mainsizer->Layout();

    Connect( idx, wxEVT_COMMAND_SPINCTRL_UPDATED,
	     wxScrollEventHandler(ROIDialog::OnROIChange),
	     NULL, NULL );
    Connect( idy, wxEVT_COMMAND_SPINCTRL_UPDATED,
	     wxScrollEventHandler(ROIDialog::OnROIChange),
	     NULL, NULL );
    Connect( idz, wxEVT_COMMAND_SPINCTRL_UPDATED,
	     wxScrollEventHandler(ROIDialog::OnROIChange),
	     NULL, NULL );

    mainsizer->SetSizeHints(this);
    mainsizer->Layout();
    // this->SetMinSize(size);
    // this->SetSize(size);

    this->handler = new ROIRefreshHandler(this);
    APP->SetRefresh2DHandler(this->handler);
    APP->Refresh2DCanvas();
  }
  
  
  ROIDialog :: ~ROIDialog(){
    int idx = x1->GetId();
    int idy = y1->GetId();
    int idz = z1->GetId();

    Disconnect( idx, wxEVT_COMMAND_SPINCTRL_UPDATED,
		wxScrollEventHandler(ROIDialog::OnROIChange),
		NULL, NULL );
    Disconnect( idy, wxEVT_COMMAND_SPINCTRL_UPDATED,
		wxScrollEventHandler(ROIDialog::OnROIChange),
		NULL, NULL );
    Disconnect( idz, wxEVT_COMMAND_SPINCTRL_UPDATED,
		wxScrollEventHandler(ROIDialog::OnROIChange),
		NULL, NULL );

    APP->SetRefresh2DHandler(NULL);
    delete this->handler;
    this->handler = NULL;
  }
  
  
  void ROIDialog::GetROI( Voxel *v0, Voxel *v1 ) {
    v0->x = x1->GetValue();
    v0->y = y1->GetValue();
    v0->z = z1->GetValue();
    v1->x = x2->GetValue();
    v1->y = y2->GetValue();
    v1->z = z2->GetValue();
  }

  void ROIDialog :: OnROIChange(wxScrollEvent& event){
    APP->Refresh2DCanvas();
  }

  void ROIDialog :: OnCancel(wxCommandEvent& event){
    //BaseDialog::OnCancel(event);
    mod->Stop();
  }

  void ROIDialog :: OnOk(wxCommandEvent& event){
    Voxel v0, v1;
    Voxel vmin, vmax;
    Voxel Cut;
    Scene *s_new = NULL;
    Scene *s_old = NULL;

    // Getting dimensions.
    GetROI( &v0, &v1 );
    vmin.x = MIN( v0.x, v1.x );
    vmin.y = MIN( v0.y, v1.y );
    vmin.z = MIN( v0.z, v1.z );
    vmax.x = MAX( v0.x, v1.x );
    vmax.y = MAX( v0.y, v1.y );
    vmax.z = MAX( v0.z, v1.z );

    // scene ROI.
    s_old = APP->Data.orig;
    s_new = ROI3( s_old, vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z );
    APP->Data.orig = s_new;
    DestroyScene( &s_old );
    MaximumValue3(APP->Data.orig);

    // Destroy Label.
    s_old = APP->Data.label;
    s_new = ROI3( s_old, vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z );
    APP->Data.label = s_new;
    DestroyScene( &s_old );

    if( APP->GetNumberOfObjs() != 0 )
      APP->DelAllObjs( );
    
    // Destroy MS Plane.
    if( APP->Data.msp != NULL ) {
      DestroyPlane( &APP->Data.msp );
      APP->Data.msp = NULL;
      APP->Data.aligned = 0;
    }

    // Arc weight ROI.
    s_old = APP->Data.arcw;
    if( s_old != NULL ) {
      s_new = ROI3( s_old, vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z );
      APP->Data.arcw = s_new;
      DestroyScene( &s_old );
    }

    // Object map weight ROI.
    s_old = APP->Data.objmap;
    if( s_old != NULL ) {
      s_new = ROI3( s_old, vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z );
      APP->Data.objmap = s_new;
      DestroyScene( &s_old );
    }

    // Gradient weight ROI.
    if( APP->Data.grad != NULL ) {
      s_old = APP->Data.grad->Gx;
      if( s_old != NULL ) {
	s_new = ROI3( s_old, vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z );
	APP->Data.grad->Gx = s_new;
	DestroyScene( &s_old );
      }
      s_old = APP->Data.grad->Gy;
      if( s_old != NULL ) {
	s_new = ROI3( s_old, vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z );
	APP->Data.grad->Gy = s_new;
	DestroyScene( &s_old );
      }
      s_old = APP->Data.grad->Gz;
      if( s_old != NULL ) {
	s_new = ROI3( s_old, vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z );
	APP->Data.grad->Gz = s_new;
	DestroyScene( &s_old );
      }
      s_old = APP->Data.grad->mag;
      if( s_old != NULL ) {
	s_new = ROI3( s_old, vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z );
	APP->Data.grad->mag = s_new;
	DestroyScene( &s_old );
      }
    }

    // Setting data dimensions.
    APP->Data.w = vmax.x - vmin.x + 1;
    APP->Data.h = vmax.y - vmin.y + 1;
    APP->Data.nframes = vmax.z - vmin.z + 1;

    // Realloc seeds
    APP->ReallocSeedStructures();

    // Set view.
    Cut.x = APP->Data.w/2;
    Cut.y = APP->Data.h/2;
    Cut.z = APP->Data.nframes/2;
    APP->Set2DViewSlice(Cut);

    //APP->Refresh2DCanvas();
    mod->Stop();
  }

} //end ModROI namespace
