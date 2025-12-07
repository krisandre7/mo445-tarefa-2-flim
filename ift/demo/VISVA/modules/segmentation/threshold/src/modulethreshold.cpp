
#include "modulethreshold.h"
#include "thresholddialog.h"


namespace ModThreshold{

  ModuleThreshold :: ModuleThreshold()
    : SegmentationModule(){
    SetName((char *)"Thresholding");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);

    TDialog = NULL;
    obj_color = 0xffff00;
    obj_name[0] = '\0';
    id1 = APP->idManager->AllocID();
    id2 = APP->idManager->AllocID();
    id3 = APP->idManager->AllocID();
  }


  ModuleThreshold :: ~ModuleThreshold(){
    APP->idManager->FreeID(id1);
    APP->idManager->FreeID(id2);
    APP->idManager->FreeID(id3);
  }

  void ModuleThreshold :: Start(){
    int r;

    APP->EnableObjWindow(false);
    APP->ResetData();

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);

    r = APP->ShowNewObjDialog(&this->obj_color,
			      this->obj_name);
    APP->SetLabelColour(0, NIL);
    APP->SetLabelColour(1, this->obj_color);

    if(TDialog!=NULL) delete TDialog;
    TDialog = new ThresholdDialog(APP->Window, id1, id2, id3, 0, 
				  MaximumValue3(APP->Data.orig),
				  this);
    TDialog->Show(true);
  }


  bool ModuleThreshold :: Stop(){
    if(TDialog!=NULL) TDialog->Show(false);
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    if(TDialog!=NULL) delete TDialog;
    TDialog = NULL;

    APP->ResetData();
    APP->DrawSegmObjects();

    return true;
  }


} //end Threshold namespace




