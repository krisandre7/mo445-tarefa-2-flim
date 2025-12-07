
#include "modulecombine.h"
//#include "combineopt.h"
#include "combinedialog.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Combine{

  ModuleCombine :: ModuleCombine() : SegmentationModule(){
    SetName((char *)"Combine");
    SetAuthor((char *)"Fábio A. M. Cappabianco");
    SoftwareVersion ver(2,0,0);
    SetVersion(ver);

    optDialog = NULL;

    obj_sel = -1;
    bkg_sel = -1;
    nobjs = 0;
    myobj = NULL;
  }


  ModuleCombine :: ~ModuleCombine(){
  }

  void ModuleCombine :: Start(){
    SegmObject *obj=NULL;
    SegmObject *bkg=NULL;
    Scene *label;
    int o, p, n;

    APP->EnableObjWindow(false);
    APP->ResetData();
    APP->SetLabelColour( 0, NIL );
    myobj = ( SegmObject* ) calloc( MAX_COBJS, sizeof( SegmObject ) );
    if( myobj == NULL )
      Error( ( char* ) "Could not alloc objects.", ( char* ) "ModuleCombine" );
    
    n = APP->Data.label->n;
    this->active = true;
    this->nobjs = APP->GetNumberOfObjs();
    
    for( o = 0; o < this->nobjs; o++ ) {
      obj = APP->GetObjByIndex( o );

      myobj[ o ].mask = BMapNew( n );
      myobj[ o ].color = obj->color;
      myobj[ o ].visibility = true;
      strcpy( myobj[ o ].name, obj->name );
      for( p = 0; p < n; p++ )
	_fast_BMapSet( myobj[ o ].mask , p, _fast_BMapGet( obj->mask, p ) );
    }

    if( this->nobjs > 0 ) {
      obj = APP->SearchObjByName( myobj[ 0 ].name );
      this->obj_sel = 0;
    }
    else {
      obj = NULL;
      this->obj_sel = -1;
    }
    if( this->nobjs > 1 ) {
      bkg = APP->SearchObjByName( myobj[ 1 ].name );
      this->bkg_sel = 1;
    }
    else {
      bkg = NULL;
      this->bkg_sel = -1;
    }

    label = APP->Data.label;
    if( ( obj != NULL ) && ( bkg != NULL ) ) {
      APP->SetLabelColour( 1, obj->color );
      APP->SetLabelColour( 2, bkg->color );
      APP->SetLabelColour( 3, obj->color ^ bkg->color );
      for( p = 0; p < label->n; p++ )
	label->data[ p ] = _fast_BMapGet( obj->mask, p ) + ( _fast_BMapGet( bkg->mask, p ) * 2 );
    }
    else if( obj != NULL ) {
      APP->SetLabelColour( 1, obj->color );
      for( p = 0; p < label->n; p++ )
	label->data[ p ] = _fast_BMapGet( obj->mask, p );
    }
    this->AllocData();
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);
  }


  void ModuleCombine :: AllocData(){
    int x,y,w,h;
    if(optDialog!=NULL) delete optDialog;
    CombineDialog *dialog = new CombineDialog(APP->Window, this);
    optDialog = (BaseDialog*)dialog;
    optDialog->Show(true);
    APP->Window->GetPosition(&x, &y);
    APP->Window->GetSize(&w, &h);
    optDialog->Move(MAX(x-20,0), h/2); //wxDefaultCoord);
  }



  bool ModuleCombine :: Stop(){
    static const char *title = {"Keep segmentation?"};
    static const char *msg = {
      "You are about to leave the combine module.\nSave changes?"};

    if(!this->active) return true;

    wxString wxtitle(title, wxConvUTF8);
    wxString wxmsg(msg, wxConvUTF8);

    wxMessageDialog dialog(APP->Window,
			   wxmsg, wxtitle,
			   wxYES_NO | wxICON_QUESTION, 
			   wxDefaultPosition);

    if(dialog.ShowModal() == wxID_YES)
      this->Finish();
    else
      this->FreeData();
    
    return true;
  }


  void ModuleCombine :: FreeData(){
    //APP->DetachOptPanel(optPanel);
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();
    this->active = false;

    optDialog->Show(false);
    if(optDialog!=NULL)
      optDialog->Destroy(); //delete optDialog;
    optDialog = NULL;
    int s;
    for( s = 0; s < nobjs; s++ )
      free( myobj[ s ].mask );
    free( myobj );
  }


  void ModuleCombine :: Finish(){
    int n, i;
    SegmObject *so;

    n = APP->Data.w*APP->Data.h*APP->Data.nframes;

    // set visibility
//     for(i=0; i<APP->GetNumberOfObjs(); i++){
//       so = APP->GetObjByIndex(i);
//       so->visibility=false;
//     }

    for( i = 0; i < nobjs; i++ ) {
      so = NULL;
      so = APP->SearchObjByName( myobj[ i ].name );      
      if( so == NULL ) {
	so = CreateSegmObject( myobj[ i ].name, myobj[ i ].color);
	APP->AddCustomObj( so );
      }
      so->seed = NULL;
      so->mask = BMapNew( myobj[ i ].mask->N );
      BMapCopy( so->mask, myobj[ i ].mask );

      if( i == 0 )
	APP->SetObjVisibility( so->name, true );
      else
	APP->SetObjVisibility( so->name, false );
    }
    this->FreeData();
  }



  void ModuleCombine :: Reset(){
    wxMessageDialog dialog(APP->Window, 
			   _T("Current segmentation will be lost.\nAre you sure you want to reset?"), 
			   _T("Reset segmentation?"), 
			   wxYES_NO | wxICON_QUESTION, 
			   wxDefaultPosition);

    if(dialog.ShowModal() == wxID_YES){
      APP->ResetData();
      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
    }
  }


  AdjRel *ModuleCombine :: GetBrush(){
    CombineDialog *opt = (CombineDialog *)optDialog;
    return opt->GetBrush();
  }

  wxCursor *ModuleCombine :: GetBrushCursor(int zoom){
    CombineDialog *opt = (CombineDialog *)optDialog;
    return opt->GetBrushCursor(zoom);
  }

  void   ModuleCombine :: NextBrush(){
    CombineDialog *opt = (CombineDialog *)optDialog;
    opt->NextBrush();
  }

  void   ModuleCombine :: PrevBrush(){
    CombineDialog *opt = (CombineDialog *)optDialog;
    opt->PrevBrush();
  }

  void ModuleCombine :: DeleteObj(int obj){
    int o;
    bool change_label;

    for(o=obj ; o<this->nobjs-1; o++){
      BMapCopy( myobj[ o ].mask, myobj[ o + 1 ].mask );
      myobj[ o ].color = myobj[ o + 1 ].color;
      myobj[ o ].visibility = myobj[ o ].visibility;
      strcpy( myobj[ o ].name, myobj[ o + 1 ].name );
    }
    this->nobjs--;
    change_label = false;
    if( bkg_sel > nobjs - 1 ) {
      this->bkg_sel--;
      change_label = true;
    }
    if( obj_sel > nobjs - 1 ) {
      this->obj_sel--;
      change_label = true;
    }
    
    if( change_label )
      ( ( CombineDialog* ) optDialog )->Refresh( );
  }

} //end Combine namespace




