
#include "moduleinterpolation.h"


namespace Interpolation{

  ModuleInterpolation :: ModuleInterpolation()
    : PreProcModule(){
    SetName((char *)"Interpolation");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleInterpolation :: ~ModuleInterpolation(){}


  void ModuleInterpolation :: Start(){
    Scene *scn = APP->Data.orig;
    float dx,dy,dz;
    int code;

    wxWindowDisabler disableAll;
    wxTheApp->Yield();

    if(scn->dx!=scn->dy || 
       scn->dx!=scn->dz || 
       scn->dy!=scn->dz){
      IsotropicDialog iso;
      code = iso.ShowModal();
      if(code==wxID_YES){
	APP->Busy((char *)"Please wait, working...");
	APP->StatusMessage((char *)"Please wait - Computation in progress...");

	scn = BIA_InterpScene2Isotropic(scn,APP->segmobjs);
	APP->SetDataVolume_NoDestroy(scn);
	APP->ReallocSeedStructures();
	APP->DrawSegmObjects();

	APP->Refresh2DCanvas();
	APP->Refresh3DCanvas(true, 1.0);
      }
    }
    else{
      CustomDialog cus;
      code = cus.ShowModal();
      if(code==wxID_OK){
	APP->Busy((char *)"Please wait, working...");
	APP->StatusMessage((char *)"Please wait - Computation in progress...");

	cus.GetVoxelSize(&dx, &dy, &dz);
	scn = BIA_LinearInterp(scn,dx,dy,dz,APP->segmobjs);
	APP->SetDataVolume_NoDestroy(scn);
	APP->ReallocSeedStructures();
	APP->DrawSegmObjects();

	APP->Refresh2DCanvas();
	APP->Refresh3DCanvas(true, 1.0);
      }
    }
    APP->StatusMessage((char *)"Done.");
    APP->Unbusy();
  }

  bool ModuleInterpolation :: Stop(){
    return true;
  }


} //end Interpolation namespace

