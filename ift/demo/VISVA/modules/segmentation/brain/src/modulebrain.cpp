
#include "modulebrain.h"


namespace Brain{

  ModuleBrain :: ModuleBrain()
    : SegmentationModule(){
    SetName((char *)"Brain: Skull Stripping");
    SetAuthor((char *)"Paulo A. V. Miranda");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }

  ModuleBrain :: ~ModuleBrain(){}

  void ModuleBrain :: Start(){
	this->Automatic();
  }


  bool ModuleBrain :: Stop(){
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();
    return true;
  }


  bool ModuleBrain :: Automatic(){
    Scene *bin,*closed;
    int color;

    APP->EnableObjWindow(false);
    //APP->ResetData();
    //APP->Refresh2DCanvas();
    //APP->Refresh3DCanvas(true,1.0);
    
    int n;
    n = APP->ShowObjColorDialog(&color,(char *)"Brain");
    if (n!=0) {
      return 0;
    }

    if(color!=NIL)
      APP->SetLabelColour(1,color);
    else
      APP->SetLabelColour(1,0xffff00);
    APP->SetLabelColour(0,NIL);

    this->Run();

    APP->AddObj((char *)"Brain",
		APP->GetLabelColour(1));

    APP->Busy((char *)"Please wait, computing Brain Envelope...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");

    bin = AddFrame3(APP->Data.label, 20, 0);
    DestroyScene(&APP->Data.label);
    closed = CloseBin3(bin, 20.0);
    DestroyScene(&bin);
    APP->Data.label = RemFrame3(closed, 20);
    DestroyScene(&closed);

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);

    APP->AddObj((char *)"Brain Envelope",
		APP->GetLabelColour(1));

    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();

    APP->SetDependencyStatus((char *)"Brain",
			     Module::SEGMENTATION,
			     true);
    APP->SetDependencyStatus((char *)"Brain Envelope",
			     Module::SEGMENTATION,
			     true);
    APP->StatusMessage((char *)"Done");
    APP->Unbusy();
    return true;
  }


  void ModuleBrain :: Run(){
    timer tic;
    Scene *cost,*pred,*scn,*bin;
    Set *S = NULL;
    Set *delSet = NULL;
    Set *B = NULL;
    int T,n,p;

    scn = APP->Data.orig;
    if(APP->Data.arcw==NULL ||
       MaximumValue3(APP->Data.arcw)==0){

      APP->Busy((char *)"Please wait, computing gradient...");
      APP->StatusMessage((char *)"Please wait - Computing Gradient...");

      //StartTimer(&tic);
      if(APP->Data.arcw!=NULL)
	DestroyScene(&APP->Data.arcw);
      APP->Data.arcw = BrainGrad3(APP->Data.orig);
      //printf(" GradMax: %d\n",SceneImax(APP->Data.arcw));
      //printf("\nGradient Time: ");
      //StopTimer(&tic);

      APP->StatusMessage((char *)"Done");
      APP->Unbusy();
    }

    APP->Busy((char *)"Please wait, computing Brain...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");
    StartTimer(&tic);

    cost = CreateScene(APP->Data.w,
		       APP->Data.h,
		       APP->Data.nframes);
    pred = CreateScene(APP->Data.w,
		       APP->Data.h,
		       APP->Data.nframes);
    SetScene(cost, INT_MAX);
    SetScene(pred,     NIL);

    n   = APP->Data.w*APP->Data.h*APP->Data.nframes;
    T   = ComputeOtsu3(scn);
    bin = Threshold3(scn, T, INT_MAX);
    APP->Data.label = ErodeBin3(bin, &S, 5.0);
    SelectLargestComp(APP->Data.label);
    DestroyScene(&bin);
    DestroySet(&S);
    B = SceneBorder(scn);
    for(p=0; p<n; p++){
      if((APP->Data.label)->data[p]>0)
	APP->AddSeed(p, 1, 1);
    }
    APP->AddSeedsInSet(B, 0, 2);
    DestroySet(&B);

    S = APP->CopySeeds();
    RunDIFT(APP->Data.arcw, cost, 
	    pred, APP->Data.label,
	    &S, &delSet);

    DestroySet(&S);
    DestroyScene(&cost);
    DestroyScene(&pred);

    printf("\nSegmentation Time: ");
    StopTimer(&tic);

    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);

    APP->StatusMessage((char *)"Done");
    APP->Unbusy();
  }


} //end Brain namespace


