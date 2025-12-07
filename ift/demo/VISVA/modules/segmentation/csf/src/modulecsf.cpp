
#include "modulecsf.h"


namespace CSF{

  ModuleCSF :: ModuleCSF()
    : SegmentationModule(){
    SetName((char *)"Brain: Cerebrospinal fluid");
    SetAuthor((char *)"Alexandre Xavier Falcao");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
    sprintf(obj_name,"CSF");
    brain = NULL;
  }

  ModuleCSF :: ~ModuleCSF(){}

  void ModuleCSF :: Start(){}


  bool ModuleCSF :: Stop(){
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();
    if(brain!=NULL)
      DestroyScene(&brain);
    return true;
  }

  void ModuleCSF :: Finish(){
    APP->AddObj(obj_name,
		APP->GetLabelColour(1));
    this->Stop();
  }

  bool ModuleCSF :: Automatic(){
    SegmObject *segm=NULL;
    int color;

    APP->Busy((char *)"Please wait, working...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");

    APP->EnableObjWindow(false);
    APP->ResetData();
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true,1.0);

    if(!APP->CheckDependency((char *)"Brain Envelope",
			     Module::SEGMENTATION)){
      if(!APP->SolveDependency((char *)"Brain Envelope",
			       Module::SEGMENTATION)){
	this->Stop();
	return false;
      }
    }

    segm = APP->SearchObjByName((char *)"Brain Envelope");
    if(segm==NULL){
      this->Stop();
      return false;
    }

    APP->Busy((char *)"Please wait, working...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");

    brain = CopyScene(APP->Data.orig);
    CopyBMap2SceneMask(brain, segm->mask);

    APP->ShowObjColorDialog(&color,(char *)"CSF");
    if(color!=NIL)
      APP->SetLabelColour(1,color);
    else
      APP->SetLabelColour(1,0xffff00);
    APP->SetLabelColour(0,NIL);

    this->Run();
    this->Finish();
    APP->SetDependencyStatus((char *)"CSF",
			     Module::SEGMENTATION,
			     true);
    return true;
  }


  void ModuleCSF :: Run(){
    timer tic;
    //Subgraph *sg=NULL;
    Scene *csf,*scn;
    int T;

    scn = APP->Data.orig;
    APP->Busy((char *)"Please wait, computing CSF...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");

    StartTimer(&tic);
    /*
    T = CSFThreshold(scn, brain);
    sg = ThresSampl3(scn, brain, T, 400);
    MarkovFeat3(sg,scn,brain,0.0);
    BestkMinCut(sg,20);    
    UnsupTrain(sg);
    Force2ClustersCSF(sg,scn,brain,T);
    csf = SceneCluster(sg,scn,brain);
    DestroyScene(&(APP->Data.label));
    APP->Data.label = csf;
    DestroySubgraph(&sg);
    */

    Scene *bin;
    Curve *hist;
    //T   = ComputeOtsu3(scn);
    hist = HistogramMask3(scn, brain);
    T   = OtsuHistogramThreshold(hist);
    bin = Threshold3(scn, 0, T);
    csf = And3(bin, brain);
    DestroyScene(&bin);
    DestroyScene(&(APP->Data.label));
    APP->Data.label = csf;
    DestroyCurve(&hist);

    printf("\nSegmentation Time: ");
    StopTimer(&tic);

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);

    APP->StatusMessage((char *)"Done");
    APP->Unbusy();
  }


} //end CSF namespace


