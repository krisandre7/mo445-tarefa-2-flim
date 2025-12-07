
#include "modulemsp.h"

namespace MSP{

  ModuleMSP :: ModuleMSP()
    : PreProcModule(){
    SetName((char *)"Mid-sagittal Plane Alignment");
    SetAuthor((char *)"Felipe P.G. Bergo");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleMSP :: ~ModuleMSP(){}


  void ModuleMSP :: Start(){
    Scene *scn = APP->Data.orig;
    VolumeOrientation ori;
    timer tic;
    char command[2048];
    int code;

    wxWindowDisabler disableAll;
    wxTheApp->Yield();

    MSPDialog dia;
    code = dia.ShowModal();
    if(code==wxID_OK){

      APP->Busy((char *)"Please wait, working...");
      APP->StatusMessage((char *)"Please wait - Computation in progress...");

      ori = dia.GetOrientation();

      WriteScene(scn,(char *)"input.scn");
      sprintf(command,"align ");
      if(ori==SagittalOrientation)
	strcat(command,"-sag");
      else if(ori==CoronalOrientation)
	strcat(command,"-cor");
      else if(ori==AxialOrientation)
	strcat(command,"-axi");
      strcat(command," input.scn output.scn");

      StartTimer(&tic);
      msp_align(command);
      printf("MSP Time: ");
      StopTimer(&tic);

      scn = ReadScene((char *)"output.scn");
      APP->SetDataVolume(scn);
      system("rm input.scn output.scn -f");

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
      APP->StatusMessage((char *)"Done.");
      APP->Unbusy();
    }
  }

  bool ModuleMSP :: Stop(){
    return true;
  }

} //end MSP namespace

