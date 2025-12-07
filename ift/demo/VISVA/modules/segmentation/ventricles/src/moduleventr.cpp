#include "moduleventr.h"
#include "ventropt.h"

namespace Ventr{

  ModuleVentr :: ModuleVentr()
    : SegmentationModule(){
    SetName((char *)"Brain: Ventricles");
    SetAuthor((char *)"Alexandre Xavier Falcao");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
    //segm_mode=0;
    sprintf(obj_name,"Lateral Ventricles");
    csf = NULL;
    VentrOpt *panel = new VentrOpt(APP->Window,this);
    optPanel = (wxPanel*)panel;
  }

  ModuleVentr :: ~ModuleVentr(){
    delete optPanel;
  }

  void ModuleVentr :: Start(){
    SegmObject *ocsf=NULL;
    int color;

    APP->EnableObjWindow(false);
    APP->ResetData();
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);

    if(!APP->CheckDependency((char *)"CSF",
			     Module::SEGMENTATION)){
      if(!APP->SolveDependency((char *)"CSF",
			       Module::SEGMENTATION)){
	wxMessageBox(_T("You must segment the CSF first."), 
		     _T("Warning"),
		     wxOK | wxICON_EXCLAMATION, APP->Window);
	this->Stop();
	return;
      }
    }

    ocsf = APP->SearchObjByName((char *)"CSF");
    if(ocsf==NULL){
      this->Stop();
      return;
    }

    APP->Busy((char *)"Please wait, working...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");

    CopyBMap2SceneMask(APP->Data.label,ocsf->mask);

    csf = CopyScene(APP->Data.label);
    Sum3inplace(APP->Data.label, csf);

    APP->ShowObjColorDialog(&color,(char *)"Lateral Ventricles");
    if(color!=NIL && color!=ocsf->color)
      APP->SetLabelColour(1,color);
    else
      APP->SetLabelColour(1,inverseColor(ocsf->color));
    APP->SetLabelColour(2,ocsf->color);
    APP->SetLabelColour(0,NIL);
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);
    APP->AppendOptPanel(optPanel, this->GetType());

    APP->StatusMessage((char *)"Done");
    APP->Unbusy();
  }

  bool ModuleVentr :: Stop(){
    APP->DetachOptPanel(optPanel);
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();
    if(csf!=NULL)
      DestroyScene(&csf);
    return true;
  }


  void ModuleVentr :: Finish(){
    int n,p,l;
    n = APP->Data.w*APP->Data.h*APP->Data.nframes;
    for(p=0; p<n; p++){
      l = (APP->Data.label)->data[p];
      if(l==2)
	(APP->Data.label)->data[p] = 0;
    }
    APP->AddObj(obj_name,
		APP->GetLabelColour(1));
    this->Stop();
  }

  void ModuleVentr :: Reset(){
    wxMessageDialog dialog(APP->Window, 
			   _T("Current segmentation will be lost.\nAre you sure you want to reset?"), 
			   _T("Reset segmentation?"), 
			   wxYES_NO | wxICON_QUESTION, 
			   wxDefaultPosition);

    if(dialog.ShowModal() == wxID_YES){
      APP->ResetData();
      DestroyScene(&APP->Data.label);
      APP->Data.label = CopyScene(csf);
      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
    }
  }

  void ModuleVentr :: Run(){
    timer tic;
    Set *obj=NULL,*bkg=NULL;
    Scene *ventr=NULL;
    int color;

    APP->DelMarkedForRemoval();
    obj = APP->CopyInternalSeeds();
    bkg = APP->CopyExternalSeeds();

    if(obj==NULL && bkg==NULL) 
      return;

    APP->Busy((char *)"Please wait, working...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");

    StartTimer(&tic);

    DestroyScene(&(APP->Data.label));
    //ventr = EraseBackground(APP->Data.orig,csf,obj,bkg);
    ventr = WatershedMask3(APP->Data.orig,csf,obj,bkg);
    APP->Data.label = Sum3(csf, csf);
    Diff3inplace(APP->Data.label, ventr);
    DestroyScene(&ventr);

    printf("\nSegmentation Time: ");
    StopTimer(&tic);

    color = APP->GetLabelColour(2);
    APP->Refresh2DCanvas();
    APP->SetLabelColour(2,NIL);
    APP->Refresh3DCanvas(true, 1.0);
    APP->SetLabelColour(2,color);

    APP->StatusMessage((char *)"Done");
    APP->Unbusy();
  
    DestroySet(&obj);
    DestroySet(&bkg);
  }


  AdjRel *ModuleVentr :: GetBrush(){
    VentrOpt *iopt = (VentrOpt *)optPanel;
    return iopt->GetBrush();
  }

} //end Ventr namespace


