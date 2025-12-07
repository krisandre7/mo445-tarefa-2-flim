
#include "modulemsp.h"

namespace MSP{

  ModuleMSP :: ModuleMSP()
  //    : RegistrationModule(){
      : PreProcModule(){
    SetName((char *)"Brain: Mid-sagittal Plane Alignment");
    SetAuthor((char *)"Guilherme C. S. Ruppert");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleMSP :: ~ModuleMSP(){}


  void ModuleMSP :: Start(){
    Scene *scn = APP->Data.orig;

    if ((scn->dx!=scn->dy) || (scn->dx!=scn->dz) || (scn->dy!=scn->dz)) {
      wxMessageBox(_T("Volume will be interpolated to isotropic voxels!"), _T("Warning"), wxOK | wxICON_EXCLAMATION, NULL);
      scn = InterpScene2Isotropic(scn);
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
      
      out = MSP_Align(scns,input_ori,1,acu);

      APP->SetDataVolume(out);

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
      APP->StatusMessage((char *)"Done.");
      APP->Unbusy();
    }
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);


  }

  void ModuleMSP :: Stop(){}

} //end MSP namespace

