
#include "modulemsp.h"

namespace MSP{

  ModuleMSP :: ModuleMSP()
    : PreProcModule(){
    SetName((char *)"Mid-sagittal Plane Alignment");
    SetAuthor((char *)"Guilherme C. S. Ruppert");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleMSP :: ~ModuleMSP(){}


  void ModuleMSP :: Start(){
    Scene *scn = APP->Data.orig;

    if( APP->Data.aligned == 1 ) {
      if (wxMessageBox(_T("Image already aligned! Align it again?"), _T("Warning"), wxYES_NO | wxICON_EXCLAMATION, NULL)==wxYES) {
	DestroyPlane( &( APP->Data.msp ) );
	APP->Data.msp = NULL;
      }
      else
	return;
    }
    
    if ((scn->dx!=scn->dy) || (scn->dx!=scn->dz) || (scn->dy!=scn->dz)) {
      wxMessageBox(_T("It is necessary to interpolate voxels to isotropic dimensions. BIA will run it in the next window."), _T("Dependency Warning"), wxOK | wxICON_EXCLAMATION, NULL);
      int n = modManager->GetNumberModules(Module::PREPROC);
      if(n==0) Error((char *)"BIAClass",(char *)"Setorientation not loaded");
      ((PreProcModule*) modManager->GetModule(Module::PREPROC, 0))->Start();
      wxTheApp->Yield();
      scn = APP->Data.orig;
      if ((scn->dx!=scn->dy) || (scn->dx!=scn->dz) || (scn->dy!=scn->dz))
	return;
    }

    wxWindowDisabler disableAll;
    wxTheApp->Yield();

    MSPDialog dia;

    int n;
    n = dia.ShowModal();
    if(n==wxID_OK){
      APP->Busy((char *)"Please wait, working...");
      APP->StatusMessage((char *)"Please wait - Computation in progress...");

      int input_ori = dia.GetOrientation();
      int acu = dia.GetAccuracy();
      Scene *out;
      Scene *mask;
     
      mask = dia.GetObject();
      out = BIA_MSP_Align(scn,mask,input_ori,acu,APP->segmobjs,&APP->Data.msp);
      APP->Data.aligned = 1;

      APP->SetDataVolume_NoDestroy(out);

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
      APP->DrawSegmObjects();

      APP->StatusMessage((char *)"Done.");
      APP->Unbusy();
    }
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);


  }

  bool ModuleMSP :: Stop(){
    return true;
  }

} //end MSP namespace

