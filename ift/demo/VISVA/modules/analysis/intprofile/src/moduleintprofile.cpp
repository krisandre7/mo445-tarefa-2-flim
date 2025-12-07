
#include "moduleintprofile.h"
#include "linehandler.h"

namespace IntensityProfile{

  ModuleIntensityProfile :: ModuleIntensityProfile()
    : AnalysisModule(){
    SetName((char *)"Intensity Profile");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
    this->handler = NULL;
  }


  ModuleIntensityProfile :: ~ModuleIntensityProfile(){}


  void ModuleIntensityProfile :: Start(){
    LineHandler *lhandler = new LineHandler(this);

    MaximumValue3(APP->Data.arcw);

    this->handler = lhandler;
    APP->EnableObjWindow(false);
    APP->ResetData();

    APP->SetLabelColour(0, NIL);
    APP->SetLabelColour(1, 0xFFFF00, 255);
    APP->SetLabelColour(2, 0xFF0000, 255);

    APP->Set2DViewInteractionHandler(lhandler, 'x');
    APP->Set2DViewInteractionHandler(lhandler, 'y');
    APP->Set2DViewInteractionHandler(lhandler, 'z');

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);
    //BaseDialog dialog(APP->Window, (char *)"Statistics");
    //dialog.AddPanel((wxPanel *)grid);
    //dialog.ShowModal();
  }

  bool ModuleIntensityProfile :: Stop(){
    LineHandler *lhandler;

    APP->SetDefaultInteractionHandler();
    if(this->handler!=NULL){
      lhandler = (LineHandler *)this->handler;
      delete lhandler;
      this->handler = NULL;
    }
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();

    return true;
  }

} //end IntensityProfile namespace

