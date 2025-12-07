#include "modulebraincluster.h"


namespace BrainCluster{

  ModuleBrainCluster :: ModuleBrainCluster()
      : SegmentationModule(){
    SetName((char *)"Brain: Tissues (WM/GM/CSF)"); //BrainCluster Segmentation");
    SetAuthor((char *)"Fabio A.M. Cappabianco");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleBrainCluster :: ~ModuleBrainCluster(){}


  void ModuleBrainCluster :: Start(){
    Scene *scn = APP->Data.orig;
    SegmObject *so;
    SegmObject *new_so;
    int p, n;

    n = APP->Data.w*APP->Data.h*APP->Data.nframes;
    if( APP->segmobjs->n == 0 ) {
      static const char *title = {"Compute brain mask?"};
      static const char *msg = {"Do you want to compute brain mask?"};
      wxString wxtitle(title, wxConvUTF8);
      wxString wxmsg(msg, wxConvUTF8);
      wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);
      
      if(dialog.ShowModal() == wxID_YES){
	int n = modManager->GetNumberModules(Module::SEGMENTATION);
	if(n<5) Error((char *)"BIAClass",(char *)"Clouds module not loaded");
	((SegmentationModule*) modManager->GetModule(Module::SEGMENTATION, 4))->Start();
	wxTheApp->Yield();
	so = APP->SearchObjByName( (char*) "Brain" );
	scn = APP->Data.orig;
	if( so == NULL )
	  return;
      }
      else {
	wxMessageBox(_T("A mask is required to segment the tissues."), _T("No Mask"), wxOK | wxICON_EXCLAMATION, APP->Window);
	return;
      }
    }
    if( APP->Data.corrected == 0 ) {
      static const char *title = {"Compute inhomogeneity?"};
      static const char *msg = {"Do you want to correct the inhomogeneity?"};
      wxString wxtitle(title, wxConvUTF8);
      wxString wxmsg(msg, wxConvUTF8);
      wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);
      
      if(dialog.ShowModal() == wxID_YES){
	int n = modManager->GetNumberModules(Module::PREPROC);
	if( n < 3 ) Error((char *)"BIAClass",(char *)"Inhomogeneity not loaded");
	((PreProcModule*) modManager->GetModule(Module::PREPROC, 3))->Start();
	wxTheApp->Yield();
	scn = APP->Data.orig;
      }
    }

    wxWindowDisabler disableAll;
    wxTheApp->Yield();
    
    // Get Object names and colors	
    char nameCSF[50]="CSF";
    char nameWM[50]="White Matter";
    char nameGM[50]="Gray Matter";
    
    BrainClusterDialog dia;
    if(dia.ShowModal()!=wxID_OK)
      return;
    APP->Busy((char *)"Please wait, working...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");
    Scene *objCSF,*objWMGM,*objWM,*objGM;

    int processors = GetNumberOfProcessors();

    if (dia.GetCsfCheck()==1) {
      // Separate CSF from WM+GM
      APP->Busy((char *)"Please wait - Separating CSF from WM+GM...");
      APP->StatusMessage((char *)"Separating CSF from WM+GM...");
      BrainClusterSegmentation( scn, dia.GetObject(), &objCSF, &objWMGM, dia.GetSamplesCSF(), dia.GetMean_PropCSF(), (float) APP->Data.modality, 
				CSF_SEG, dia.GetAuto_PropCSF(), 0.0, dia.GetKMinCSF(), dia.GetKMaxCSF(), processors );

      // WriteScene( objCSF, "CSF_teste.scn.bz2" );
      // WriteScene( objWMGM, "WMGM_teste.scn.bz2" );
      // Reset Brain CSF
      so = APP->SearchObjByName( (char*) "Brain" );
      if( so != NULL ) {
	for( p = 0; p < n; p++ ) {
	  if( objCSF->data[ p ] > 0 )
	    _fast_BMapSet0(so->mask, p);
	}
      }
      // Reset Left Hemisphere CSF
      so = APP->SearchObjByName( ( char* ) "Left Hemisphere" );
      if( so != NULL ) {
	for( p = 0; p < n; p++ ) {
	  if( objCSF->data[ p ] > 0 )
	    _fast_BMapSet0(so->mask, p);
	}
      }
      // Reset Right Hemisphere CSF
      so = APP->SearchObjByName( ( char* ) "Right Hemisphere" );
      if( so != NULL ) {
	for( p = 0; p < n; p++ ) {
	  if( objCSF->data[ p ] > 0 )
	    _fast_BMapSet0(so->mask, p);
	}
      }
      // Reset Cerebellum CSF
      so = APP->SearchObjByName( ( char* ) "Cerebellum" );
      if( so != NULL ) {
	for( p = 0; p < n; p++ ) {
	  if( objCSF->data[ p ] > 0 )
	    _fast_BMapSet0(so->mask, p);
	}
      }

      // Separate WM from GM
      APP->Busy((char *)"Please wait - Separating WM from GM...");
      APP->StatusMessage((char *)"Separating WM from GM...");
      // printf( "samples:%.2f, Mean_Prop:%.2f, Auto_Prop:%.2f\n", dia.GetSamples(), dia.GetMean_Prop(), dia.GetAuto_Prop() );
      BrainClusterSegmentation( scn, objWMGM, &objGM, &objWM, dia.GetSamples(), dia.GetMean_Prop(), (float) APP->Data.modality, 
				WMGM_SEG, dia.GetAuto_Prop(), 0.0, dia.GetKMin(), dia.GetKMax(), processors );
      // WriteScene( objGM, "GM_teste.scn.bz2" );
      // WriteScene( objWM, "WM_teste.scn.bz2" );

      AddMaskToSegmObj(objCSF,nameCSF,dia.GetColorCSF());
      AddMaskToSegmObj(objWM,nameWM,dia.GetColorWM());
      AddMaskToSegmObj(objGM,nameGM,dia.GetColorGM());
      SegmObject *segm;
      segm = APP->SearchObjByName(nameCSF);
      segm->visibility=true;
      segm = APP->SearchObjByName(nameWM);
      segm->visibility=true;
      segm = APP->SearchObjByName(nameGM);
      segm->visibility=true;
      APP->DrawSegmObjects();
      DestroyScene(&objCSF);
      DestroyScene(&objWMGM);
    }
    else {
      // Separate WM from GM
      APP->Busy((char *)"Please wait - Separating WM from GM...");
      APP->StatusMessage((char *)"Separating WM from GM...");
      BrainClusterSegmentation( scn, dia.GetObject(), &objGM, &objWM, dia.GetSamples(), dia.GetMean_Prop(), (float) APP->Data.modality, 
				WMGM_SEG, dia.GetAuto_Prop(), 0.0, dia.GetKMin(), dia.GetKMax(), processors );
      AddMaskToSegmObj(objWM,nameWM,dia.GetColorWM());
      AddMaskToSegmObj(objGM,nameGM,dia.GetColorGM());
      SegmObject *segm;
      segm = APP->SearchObjByName(nameWM);
      segm->visibility=true;
      segm = APP->SearchObjByName(nameGM);
      segm->visibility=true;
      APP->DrawSegmObjects();
    }

    // Creating label Left Hemisphere WM and GM
    so = APP->SearchObjByName( ( char* ) "Left Hemisphere" );
    if (so!=NULL) {
      // Creating label Left Hemisphere WM
      new_so = CreateSegmObject( (char *)"Left Hemisphere WM", dia.GetColorWM() );
      new_so->mask = BMapNew( n );
      BMapFill( new_so->mask, 0 );
      for( p = 0; p < n; p++ ) {
	if ( objWM->data[ p ] > 0  && _fast_BMapGet( so->mask, p ) != 0 ) 
	  _fast_BMapSet1( new_so->mask, p );
      }
      APP->AddCustomObj( new_so );
      // Creating label Left Hemisphere GM
      new_so = CreateSegmObject( (char *)"Left Hemisphere GM", dia.GetColorGM() );
      new_so->mask = BMapNew( n );
      BMapFill( new_so->mask, 0 );
      for( p = 0; p < n; p++ ) {
	if( ( objGM->data[ p ] > 0 ) && ( _fast_BMapGet( so->mask, p ) != 0 ) )
	  _fast_BMapSet1( new_so->mask, p );
      }
      APP->AddCustomObj( new_so );
    }


    // Creating label Right Hemisphere WM and GM
    so = APP->SearchObjByName( ( char* ) "Right Hemisphere" );
    if (so!=NULL) {
      // Creating label Right Hemisphere WM
      new_so = CreateSegmObject( (char *)"Right Hemisphere WM", dia.GetColorWM() );
      new_so->mask = BMapNew( n );
      BMapFill( new_so->mask, 0 );
      for( p = 0; p < n; p++ ) {
	if( ( objWM->data[ p ] > 0 ) && ( _fast_BMapGet( so->mask, p ) != 0 ) )
	  _fast_BMapSet1( new_so->mask, p );
      }
      APP->AddCustomObj( new_so );
      
      // Creating label Right Hemisphere GM
      new_so = CreateSegmObject( (char *)"Right Hemisphere GM", dia.GetColorGM() );
      new_so->mask = BMapNew( n );
      BMapFill( new_so->mask, 0 );
      for( p = 0; p < n; p++ ) {
	if( ( objGM->data[ p ] > 0 ) && ( _fast_BMapGet( so->mask, p ) != 0 ) )
	  _fast_BMapSet1( new_so->mask, p );
      }
      APP->AddCustomObj( new_so );
    }      

    // Creating label Cerebellum WM and GM
    so = APP->SearchObjByName( ( char* ) "Cerebellum" );
    if (so!=NULL) {
      // Creating label Cerebellum WM
      new_so = CreateSegmObject( (char *)"Cerebellum WM", dia.GetColorWM() );
      new_so->mask = BMapNew( n );
      BMapFill( new_so->mask, 0 );
      for( p = 0; p < n; p++ ) {
	if( ( objWM->data[ p ] > 0 ) && ( _fast_BMapGet( so->mask, p ) != 0 ) )
	  _fast_BMapSet1( new_so->mask, p );
      }
      APP->AddCustomObj( new_so );
      
      // Creating label Cerebellum GM
      new_so = CreateSegmObject( (char *)"Cerebellum GM", dia.GetColorGM() );
      new_so->mask = BMapNew( n );
      BMapFill( new_so->mask, 0 );
      for( p = 0; p < n; p++ ) {
	if( ( objGM->data[ p ] > 0 ) && ( _fast_BMapGet( so->mask, p ) != 0 ) )
	  _fast_BMapSet1( new_so->mask, p );
      }
      APP->AddCustomObj( new_so );
    }

    if (dia.GetCsfCheck()==1) {
      so = APP->SearchObjByName((char *)"CSF");
      so->visibility=true;
    }
    so = APP->SearchObjByName((char *)"White Matter");
    so->visibility=true;
    so = APP->SearchObjByName((char *)"Gray Matter");
    so->visibility=true;
    so = APP->SearchObjByName((char *)"Brain");
    so->visibility=false;


    APP->DrawSegmObjects();

    DestroyScene(&objWM);
    DestroyScene(&objGM);

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    APP->StatusMessage((char *)"Done.");
    APP->Unbusy();
    
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    
    
  }

  bool ModuleBrainCluster :: Stop(){
    return true;
  }

  void ModuleBrainCluster :: AddMaskToSegmObj(Scene *mask, char *obj_name, int obj_color) {
    int n = APP->Data.w*APP->Data.h*APP->Data.nframes;
    SegmObject *obj = CreateSegmObject(obj_name,obj_color);
    obj->mask = BMapNew(n);
    obj->visibility=1;
    BMapFill(obj->mask, 0);
    int p;
    for(p=0; p<n; p++){
      if (mask->data[p]!=0) _fast_BMapSet1(obj->mask, p);
    }
    obj->seed = NULL;
    APP->AddCustomObj(obj);
  };


} //end BrainCluster namespace

