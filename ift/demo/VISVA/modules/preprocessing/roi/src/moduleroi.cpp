
#include "moduleroi.h"
#include "roidialog.h"


namespace ModROI{

  ModuleROI :: ModuleROI()
    : PreProcModule(){
    SetName((char *)"R.O.I.");
    SetAuthor((char *)"F.A.M. Cappabianco");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);

    TDialog = NULL;
    //obj_color = 0xffff00;
    //obj_name[0] = '\0';
    idx = APP->idManager->AllocID();
    idy = APP->idManager->AllocID();
    idz = APP->idManager->AllocID();
  }


  ModuleROI :: ~ModuleROI(){
    APP->idManager->FreeID(idx);
    APP->idManager->FreeID(idy);
    APP->idManager->FreeID(idz);
  }

  void ModuleROI :: Start(){
    //int r;

    APP->EnableObjWindow(false);
    APP->ResetData();

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);

    APP->SetLabelColour(0, NIL);
    APP->SetLabelColour(1, 0xff0000);

    if(TDialog!=NULL) delete TDialog;
    TDialog = new ROIDialog(APP->Window, idx, idy, idz, APP->Data.orig, this);
    TDialog->Show(true);
  }


  bool ModuleROI :: Stop(){
    if(TDialog!=NULL) TDialog->Show(false);
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    if(TDialog!=NULL) delete TDialog;
    TDialog = NULL;
    APP->ResetData();
    APP->DrawSegmObjects();

    return true;
  }


} //end ROI namespace

