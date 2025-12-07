
#include "modulecloudsbrain.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace CloudsBrain{

  ModuleCloudsBrain :: ModuleCloudsBrain()
    : SegmentationModule(){
    //SetName((char *)"Telencephalon/Cerebellum");
    SetName((char *)"Brain: Cerebral Hemisph./Cerebel.");
    SetAuthor((char *)"Paulo A.V. Miranda");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleCloudsBrain :: ~ModuleCloudsBrain(){
  }


  void ModuleCloudsBrain :: Start(){
    SegmObject *obj=NULL;
    Scene *label=NULL;
    int p,n;
    APP->EnableObjWindow(false);
    APP->ResetData();

    if(APP->Data.oriented == 0 ) {
      int n = modManager->GetNumberModules(Module::PREPROC);
      if(n==0) Error((char *)"BIAClass",(char *)"Setorientation not loaded");
      ((PreProcModule*) modManager->GetModule(Module::PREPROC, 1))->Start();
      wxTheApp->Yield();
      if( APP->Data.oriented == 0 )
	return;
    }

    /*    if(!APP->Data.oriented){
    wxMessageBox(_T("You must first specify the volume orientation."), 
		   _T("Volume not oriented"),
		   wxOK | wxICON_EXCLAMATION, APP->Window);
      this->Stop();
      return;
    }
    */

    APP->Busy((char *)"Please wait, working...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");

    label = CloudsBrainSegmentation(APP->Data.orig, 
				    (char *)"triobj");
				    //(char *)"brain");

    //--------------------------------
    n = APP->Data.w*APP->Data.h*APP->Data.nframes;

    /*
    obj = CreateSegmObject((char *)"Telencephalon",
			   0xffff00);
    obj->mask = BMapNew(n);
    BMapFill(obj->mask, 0);
    for(p=0; p<n; p++)
      if(label->data[p]>1)
	_fast_BMapSet1(obj->mask, p);
    APP->AddCustomObj(obj);
    */

    obj = CreateSegmObject((char *)"Left Hemisphere",
			   0xadd8e6);
    obj->mask = BMapNew(n);
    BMapFill(obj->mask, 0);
    for(p=0; p<n; p++)
      if(label->data[p]==3)
	_fast_BMapSet1(obj->mask, p);
    APP->AddCustomObj(obj);


    obj = CreateSegmObject((char *)"Right Hemisphere",
			   0x90ee90);
    obj->mask = BMapNew(n);
    BMapFill(obj->mask, 0);
    for(p=0; p<n; p++)
      if(label->data[p]==2)
	_fast_BMapSet1(obj->mask, p);
    APP->AddCustomObj(obj);


    obj = CreateSegmObject((char *)"Cerebellum",
			   0xffff00);
    obj->mask = BMapNew(n);
    BMapFill(obj->mask, 0);
    for(p=0; p<n; p++)
      if(label->data[p]==1)
	_fast_BMapSet1(obj->mask, p);
    APP->AddCustomObj(obj);


    obj = CreateSegmObject((char *)"Brain",
			   0xffff00);
    obj->mask = BMapNew(n);
    BMapFill(obj->mask, 0);
    for(p=0; p<n; p++)
      if(label->data[p]>0)
	_fast_BMapSet1(obj->mask, p);
    APP->AddCustomObj(obj);


    //obj = APP->SearchObjByName((char *)"Telencephalon");
    //obj->visibility=true;
    obj = APP->SearchObjByName((char *)"Left Hemisphere");
    obj->visibility=true;
    obj = APP->SearchObjByName((char *)"Right Hemisphere");
    obj->visibility=true;
    obj = APP->SearchObjByName((char *)"Cerebellum");
    obj->visibility=true;
    obj = APP->SearchObjByName((char *)"Brain");
    obj->visibility=false;

    //--------------------------------
    DestroyScene(&label);

    APP->StatusMessage((char *)"Done");
    APP->Unbusy();

    this->Stop();
  }


  bool ModuleCloudsBrain :: Stop(){
    APP->SetDefaultInteractionHandler();
    APP->EnableObjWindow(true);
    APP->ResetData();
    APP->DrawSegmObjects();
    return true;
  }


} //end CloudsBrain namespace




