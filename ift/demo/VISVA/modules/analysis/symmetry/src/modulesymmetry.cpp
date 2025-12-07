
#include "modulesymmetry.h"

namespace Symmetry{

  ModuleSymmetry :: ModuleSymmetry()
    : AnalysisModule(){
    SetName((char *)"EXPERIMENTAL: Symmetry Analysis");
    SetAuthor((char *)"Guilherme C. S. Ruppert");
    SoftwareVersion ver(1,2,0);
    SetVersion(ver);
  }


  ModuleSymmetry :: ~ModuleSymmetry(){}


  void ModuleSymmetry :: Start(){


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
    
    Scene *scn = APP->Data.orig;
   
    // Analysis of FP/FN of segmentation
    SegmObject *obj=NULL;
    int i;
    int n = APP->segmobjs->n;
    scn = APP->Data.orig;
    for (i=0; i < n; i++) {
      obj = (SegmObject *)GetArrayListElement(APP->segmobjs, i);
      printf("Processing object %s...\n",obj->name);
      Scene *mask = CreateScene(scn->xsize,scn->ysize,scn->zsize);
      CopyBMap2SceneMask(mask,obj->mask);
      
      Scene *out = SymmetryDiffBinScene(mask);
      char name[1024];
      sprintf(name,"%s_S",obj->name);
      obj->visibility = false;
      AddMaskToSegmObj(out,name,obj->color);
      DestroyScene(&mask);
      DestroyScene(&out);
    }
    // Analysis of CSF/GM of segmentation
    printf("Processing CSF vs. GM %s...\n",obj->name);
    SegmObject *obj1=NULL, *obj2=NULL;
    n = APP->segmobjs->n;
    scn = APP->Data.orig;
    for (i=0; i < n; i++) {
      obj1 = (SegmObject *)GetArrayListElement(APP->segmobjs, i);
      if (strcmp(obj1->name,"CSF")==0) break;
    }
    for (i=0; i < n; i++) {
      obj2 = (SegmObject *)GetArrayListElement(APP->segmobjs, i);
      if (strcmp(obj2->name,"Gray Matter")==0) break;
    }
    Scene *maskcsf = CreateScene(scn->xsize,scn->ysize,scn->zsize);
    CopyBMap2SceneMask(maskcsf,obj1->mask);
    WriteScene(maskcsf, ( char* ) "csf.scn");
    Scene *maskgm = CreateScene(scn->xsize,scn->ysize,scn->zsize);
    CopyBMap2SceneMask(maskgm,obj2->mask);
    WriteScene(maskgm, ( char* ) "gm.scn");
    Scene *res = CreateScene(scn->xsize,scn->ysize,scn->zsize);
    Voxel v;
    int p,q;
    int mid = scn->xsize/2;
    for (v.z=0; v.z < scn->zsize; v.z++) 
      for (v.y=0; v.y < scn->ysize; v.y++) 
	for (v.x=1; v.x < scn->xsize; v.x++) {
	  p = v.x + scn->tby[v.y] + scn->tbz[v.z];
	  q = (mid+mid-v.x) + scn->tby[v.y] + scn->tbz[v.z];
	  if (//(maskgm->data[p]==1 && maskcsf->data[q]==1) ||
	      (maskgm->data[q]==1 && maskcsf->data[p]==1))
	    res->data[p]=1;
	}
    AddMaskToSegmObj(res, ( char* ) "CSF_GM_Diff",obj1->color);
    DestroyScene(&res);
    DestroyScene(&maskgm);
    DestroyScene(&maskcsf);



  

    // Analysis of texture differences
    scn = APP->Data.orig;
    for (i=0; i < n; i++) {
      obj = (SegmObject *)GetArrayListElement(APP->segmobjs, i);
      if (strcmp(obj->name,"Gray Matter")==0) {
	printf("Processing object %s...\n",obj->name);
	Scene *mask = CreateScene(scn->xsize,scn->ysize,scn->zsize);
	CopyBMap2SceneMask(mask,obj->mask);

	Scene *res = CreateScene(scn->xsize,scn->ysize,scn->zsize);
	Voxel u;
	int p,q;
	int mid = scn->xsize/2;
	for (u.z=0; u.z < scn->zsize; u.z++) 
	  for (u.y=0; u.y < scn->ysize; u.y++) 
	    for (u.x=1; u.x < scn->xsize; u.x++) {
	      p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	      q = (mid+mid-u.x) + scn->tby[u.y] + scn->tbz[u.z];
	      if (mask->data[p]==1 || mask->data[q]==1) 
		res->data[p]=abs(scn->data[p]-scn->data[q]);
	    }
	char name[1024];
	sprintf(name,"%s_GM_diff.scn",APP->Data.projectfilename);
	WriteScene(res,name);
	DestroyScene(&res);
	DestroyScene(&mask);
      }
    }

    APP->DrawSegmObjects();
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    APP->StatusMessage((char *)"Done.");
    APP->Unbusy();
  }


  Scene* ModuleSymmetry :: SymmetryDiffBinScene(Scene *scn) {

    Voxel u;
    int p,q;
    Scene *res = CreateScene(scn->xsize,scn->ysize,scn->zsize);
    /* // simetrico
    int mid = scn->xsize/2;
    for (u.z=0; u.z < scn->zsize; u.z++) 
      for (u.y=0; u.y < scn->ysize; u.y++) 
	for (u.x=scn->xsize/2+1; u.x < scn->xsize; u.x++) {
	  p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	  q = (mid-(u.x-mid)) + scn->tby[u.y] + scn->tbz[u.z];
	  if (scn->data[p]!=scn->data[q]) {
	    res->data[p]=1;
	    res->data[q]=1;
	  }
	} 
    */
    int mid = scn->xsize/2;
    for (u.z=0; u.z < scn->zsize; u.z++) 
      for (u.y=0; u.y < scn->ysize; u.y++) 
	for (u.x=1; u.x < scn->xsize; u.x++) {
	  if (u.x==mid) continue;
	  p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	  q = (mid+mid-u.x) + scn->tby[u.y] + scn->tbz[u.z];
	  if (scn->data[p]==1 && scn->data[q]==0)
	    res->data[p]=1;
	  if (scn->data[p]==0 && scn->data[q]==1)
	    res->data[q]=1;
	} 
    return res;
  }


  void ModuleSymmetry :: AddMaskToSegmObj(Scene *mask, char *obj_name, int obj_color) {
    int n = APP->Data.w*APP->Data.h*APP->Data.nframes;
    SegmObject *obj = CreateSegmObject(obj_name,obj_color);
    obj->mask = BMapNew(n);
    obj->visibility=1;
    BMapFill(obj->mask, 0);
    int p;
    for(p=0; p<n; p++){
      if (mask->data[p]!=0) _fast_BMapSet1(obj->mask, p);
    }
    obj->seed = NULL;
    APP->AddCustomObj(obj);
  };


} //end Symmetry namespace

