
#include "moduleselfreg.h"

namespace SelfReg{

  ModuleSelfReg :: ModuleSelfReg()
  : RegistrationModule(){
    SetName((char *)"Self Registration");
    SetAuthor((char *)"Guilherme C. S. Ruppert");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleSelfReg :: ~ModuleSelfReg(){}


  void ModuleSelfReg :: Start(){
  
    if(APP->Data.oriented == 0 ) {
      wxMessageBox(_T("Image orientation is unknown. Please specify it in the next window."),_T("Dependency Warning"),wxOK | wxICON_EXCLAMATION, APP->Window);
      int n = modManager->GetNumberModules(Module::PREPROC);
      if(n==0) Error((char *)"BIAClass",(char *)"Setorientation not loaded");
      ((PreProcModule*) modManager->GetModule(Module::PREPROC, 1))->Start();
      wxTheApp->Yield();
      if( APP->Data.oriented == 0 )
	return;
    }

    if(APP->Data.aligned == 0 ) {
      wxMessageBox(_T("Mid-sagittal plane alignment needs to be ran first. BIA is going to run it in the next window."),_T("Dependency Warning"),wxOK | wxICON_EXCLAMATION, APP->Window);
      int n = modManager->GetNumberModules(Module::PREPROC);
      if(n==0) Error((char *)"BIAClass",(char *)"MSP Alignment not loaded");
      ((PreProcModule*) modManager->GetModule(Module::PREPROC, 2))->Start();
      wxTheApp->Yield();
      if( APP->Data.aligned == 0 )
	return;
    }
    
    APP->Busy((char *)"Please wait, working...");
    APP->StatusMessage((char *)"Please wait - Computation in progress...");
    float T[4][4];
    float *best_theta;
    Scene *scn = APP->Data.orig;
    Scene *out = SelfRegisterAxial(scn,T,&best_theta);

    // Show registered scene
    Scene *left,*right,*rightf;
    left = GetLeftAxialHalf(out);
    right = GetRightAxialHalf(out);
    rightf = FlipSceneAxial(right);
    DestroyScene(&right);
    Scene *R,*G,*B;
    ColorReg(left,rightf,&R,&G,&B);
    SelfRegDialog *dia = new SelfRegDialog(R,G,B);
    if (dia->ShowModal()!=wxID_OK) {
      DestroyScene(&R);
      DestroyScene(&G);
      DestroyScene(&B);
      DestroyScene(&left);
      DestroyScene(&rightf);
      APP->StatusMessage((char *)"Done.");
      APP->Unbusy();
      return;
    }
    DestroyScene(&R);
    DestroyScene(&G);
    DestroyScene(&B);

    APP->SetDataVolume_NoDestroy(out);

    SegmObject *obj=NULL;
    int i;
    int n = APP->segmobjs->n;
    scn = APP->Data.orig;
    for (i=0; i < n; i++) {
      obj = (SegmObject *)GetArrayListElement(APP->segmobjs, i);
      printf("Transforming object %s...\n",obj->name);
      Scene *mask1 = CreateScene(scn->xsize,scn->ysize,scn->zsize);
      CopyBMap2SceneMask(mask1,obj->mask);

      // Trasnforming the objects
      Scene *right,*rightf,*newrightf,*newright;
      right = GetRightAxialHalf(mask1);
      rightf = FlipSceneAxial(right);
      DestroyScene(&right);
      newrightf=CreateScene(rightf->xsize,rightf->ysize,rightf->zsize);      
      //transformScene(rightf,T,newrightf);
      MytransformScene(rightf,T,newrightf);
      DestroyScene(&rightf);
      newright=FlipSceneAxial(newrightf);
      DestroyScene(&newrightf);
      Scene *res = CopyScene(mask1);
      Voxel u;
      int p,q;
      for (u.z=0; u.z < res->zsize; u.z++)
	for (u.y=0; u.y < res->ysize; u.y++)
	  for (u.x=(res->xsize/2)+1; u.x < res->xsize; u.x++) {
	    p = u.x + res->tby[u.y] + res->tbz[u.z];
	    q = u.x-res->xsize/2-1 + newright->tby[u.y] + newright->tbz[u.z];
	    res->data[p]=newright->data[q];
	  }
      DestroyScene(&newright);
      DestroyScene(&mask1);
      BMap *newbmap;
      newbmap = SceneMask2BMap(res);
      DestroyScene(&res);
      BMapCopy(obj->mask,newbmap);
      BMapDestroy(newbmap);      
    }


    APP->StatusMessage((char *)"Done.");
    APP->Unbusy();

    wxWindowDisabler disableAll;
    wxTheApp->Yield();
  
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);  
    APP->DrawSegmObjects();
  }

  bool ModuleSelfReg :: Stop(){
    return true;
  }


  


} //end SelfReg namespace
