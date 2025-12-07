
#include "modsegexternal.h"


namespace SegExternal{

  ModSegExternal :: ModSegExternal(char *name, char *author)
    : SegmentationModule(){
    SetName((char *)name);
    SetAuthor((char *)author);
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }

  ModSegExternal :: ~ModSegExternal(){}

  void ModSegExternal :: Start(){
    Scene *label=NULL;
    SegmObject *obj=NULL;
    int n,p;
    timer tic;
    
    n = APP->Data.w*APP->Data.h*APP->Data.nframes;

    APP->ResetData();
    APP->DrawSegmObjects();

    APP->EnableObjWindow(false);
    APP->Busy((char *)"Please wait, computing...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");

    WriteScene(APP->Data.orig,(char *)"image.hdr");

    //----------------------------------------------
    StartTimer(&tic);

    system("./bet.sh image.hdr");
    //system("rm image.hdr image.img");

    printf("\n\tExternal Time: ");
    StopTimer(&tic);
    //----------------------------------------------

    label = ReadScene((char *)"image_mask.scn.bz2");

    obj = CreateSegmObject((char *)"Brain", 0xffcccc);
    obj->mask = BMapNew(n);
    BMapFill(obj->mask, 0);
    for(p=0; p<n; p++){
      if(label->data[p]>0)
	_fast_BMapSet1(obj->mask, p);
    }
    obj->seed = NULL;
    APP->AddCustomObj(obj);
    APP->SetObjVisibility(obj->name, true);

    DestroyScene(&label);

    APP->StatusMessage((char *)"Done");
    APP->Unbusy();
    APP->EnableObjWindow(true);
    APP->DrawSegmObjects();
  }


  bool ModSegExternal :: Stop(){
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->DrawSegmObjects();
    return true;
  }


} //end SegExternal namespace
