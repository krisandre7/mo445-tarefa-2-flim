

#include "moduleinhomogeneity.h"


namespace Inhomogeneity{

  ModuleInhomogeneity :: ModuleInhomogeneity()
  //    : RegistrationModule(){
      : PreProcModule(){
    SetName((char *)"Brain: Inhomogeneity Correction");
    SetAuthor((char *)"Fabio A.M. Cappabianco");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleInhomogeneity :: ~ModuleInhomogeneity(){}


  void ModuleInhomogeneity :: Start(){
    Scene *scn = APP->Data.orig;
    int n;
    SegmObject *so;
    //char objname[ 1024 ];

    //if ( n == 0 ) {
    //wxMessageBox(_T("Brain object is required!"), _T("Warning"), wxOK | wxICON_EXCLAMATION, NULL);
    //return;
    //}
    //strcpy( objname, "Brain" );
    if( APP->Data.corrected == 1 ) {

      static const char *title = {"Compute inhomogeneity?"};
      static const char *msg = {"Inhomogeneity was already corrected. Proceed anyway?"};
      wxString wxtitle(title, wxConvUTF8);
      wxString wxmsg(msg, wxConvUTF8);
      wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);
      
      if(dialog.ShowModal() == wxID_NO)
	return;
    }

    //so = APP->SearchObjByName( (char*) "Brain" );
    //if( so == NULL ) {
    n = APP->segmobjs->n;
    if ( n == 0 ) {
      static const char *title = {"No object found?"};
      static const char *msg = {"There is no object to be corrected. Compute brain mask?"};
      wxString wxtitle(title, wxConvUTF8);
      wxString wxmsg(msg, wxConvUTF8);
      wxMessageDialog dialog(APP->Window, wxmsg, wxtitle, wxYES_NO | wxICON_QUESTION, wxDefaultPosition);
      if(dialog.ShowModal() == wxID_NO)
	return;
      int n = modManager->GetNumberModules(Module::SEGMENTATION);
      if(n<5) Error((char *)"BIAClass",(char *)"Clouds module not loaded");
      ((SegmentationModule*) modManager->GetModule(Module::SEGMENTATION, 4))->Start();
      wxTheApp->Yield();
      so = APP->SearchObjByName( (char*) "Brain" );
      scn = APP->Data.orig;
      if( so == NULL )
	return;
    }
 
    /*
    if ((scn->dx!=scn->dy) || (scn->dx!=scn->dz) || (scn->dy!=scn->dz)) {
      wxMessageBox(_T("Volume will be interpolated to isotropic voxels!"), _T("Warning"), wxOK | wxICON_EXCLAMATION, NULL);
      scn = InterpScene2Isotropic(scn);
    }
    */

    //wxWindowDisabler disableAll;
    wxTheApp->Yield();
    InhomogeneityDialog dia;

    n = dia.ShowModal();
    if(n==wxID_OK){
      APP->Busy((char *)"Please wait, working...");
      APP->StatusMessage((char *)"Please wait - Computation in progress...");

      Scene* mask = dia.GetObject();
      int proto = (int) APP->Data.modality; //dia.GetProtocol();
      int comp = 2; //dia.GetCompression();
      double func = dia.GetFunc();
      double radius = dia.GetRadius();
      double rgrowth = 1.6;
      int keep = 0; //dia.GetKeep();
      Scene *out;
      int processors = GetNumberOfProcessors();
      out = InhomogeneityCorrection( scn, mask, func, radius, rgrowth, comp, proto, processors, 1 );

      if (keep==0) {
	int p;
	int n=scn->xsize*scn->ysize*scn->zsize;
	for (p=0;p<n;p++) 
	  if (mask->data[p]==0) out->data[p]=scn->data[p];
      }

      DestroyScene( &mask );

      if (out==NULL) {
	wxMessageBox(_T("Module Error. Probably due to invalid parameters!"), _T("Warning"), wxOK | wxICON_EXCLAMATION, NULL);
	return;
      } 
      
      APP->SetDataVolume_NoDestroy(out);
      APP->Data.corrected = 1;

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
      APP->StatusMessage((char *)"Done.");
      APP->Unbusy();
    }
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);


  }

  bool ModuleInhomogeneity :: Stop(){
    return true;
  }

} //end Inhomogeneity namespace

