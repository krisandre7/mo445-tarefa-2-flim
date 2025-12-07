
#include "modulerigidreg.h"

namespace RigidReg{

  ModuleRigidReg :: ModuleRigidReg()
  : RegistrationModule(){
    SetName((char *)"Rigid Registration");
    SetAuthor((char *)"Guilherme C. S. Ruppert and Fernanda Favretto");
    SoftwareVersion ver(1,0,0);
    SetVersion(ver);
  }


  ModuleRigidReg :: ~ModuleRigidReg(){}


  void ModuleRigidReg :: Start(){

    Scene *scn = APP->Data.orig;

    if ((scn->dx!=scn->dy) || (scn->dx!=scn->dz) || (scn->dy!=scn->dz)) {
      if (wxMessageBox(_T("Volume will be interpolated to isotropic voxels! Continue?"), _T("Confirmation"), wxYES_NO, NULL)==wxNO)
	return;
      scn = InterpScene2Isotropic(scn);
      APP->SetDataVolume(scn);
    }

    wxWindowDisabler disableAll;
    wxTheApp->Yield();


    char *dir = (APP->Preferences).dir_LoadVolume;

  static
    wxChar
    *FILETYPES = _T( "Supported Formats (*.bia;*.scn;*.hdr;*.scn.bz2)|*.bia;*.scn;*.hdr;*.scn.bz2|"
		     "BIA files (*.bia)|*.bia|"
		     "Scene files (*.scn)|*.scn|"
		     "Analyze files (*.hdr)|*.hdr|"
		     "Compressed Scene files (*.scn.bz2)|*.scn.bz2|"
		     "All files|*.*"
		     );
    
    wxFileDialog *openFileDialog =
      new wxFileDialog ( APP->Window,
			 _T("Select the scene to register to "),
			 _T(""),
			 _T(""),
			 FILETYPES,
			 wxFD_OPEN,
			 wxDefaultPosition);
    
    openFileDialog->SetFilterIndex((APP->Preferences).ext_LoadVolume);
    if(dir[0]!='\0'){
      wxString wxdir(dir, wxConvUTF8);
      openFileDialog->SetDirectory(wxdir);
    }
    if(openFileDialog->ShowModal() == wxID_OK){ 
      (APP->Preferences).ext_LoadVolume = openFileDialog->GetFilterIndex();
      char path[512];
      strcpy( dir,  openFileDialog->GetDirectory().ToAscii() );
      strcpy( path, openFileDialog->GetPath().ToAscii() );
      //strcpy(APP->Data.volumename, openFileDialog->GetFilename().ToAscii());

      char *filename = path;
      Scene *fixed1,*fixed;
      int s = strlen(filename);
      if( strcasecmp(filename + s - 3, "scn") == 0 )
	fixed1 = ReadScene(filename);
      else if( strcasecmp(filename + s - 3, "hdr") == 0 )
	fixed1 = ReadScene(filename);
      else if( strcasecmp(filename + s - 3, "bz2") == 0 )
	fixed1 = ReadCompressedScene(filename);
      else{
	wxMessageBox(_T("Unable to load volume data. Invalid file format!"), _T("Load Volume Error"), wxOK | wxICON_EXCLAMATION, APP->Window);
	return;
      }
      if (fixed1 == NULL) {
	wxMessageBox(_T("Corrupted file!"), _T("Load Volume Error"),
		     wxOK | wxICON_EXCLAMATION, APP->Window);
	return;
      }
      
      fixed=fixed1;
      // Matching the voxel dimensions between fixed and moving images
      if ((scn->dx!=fixed1->dx) || (scn->dx!=fixed1->dy) || (scn->dy!=fixed1->dz)) {
	// Get the minimum dimension
	float min = scn->dx;
	int where = 2; // 1=fixed / 2=scn
	if (min>fixed1->dx) { min=fixed1->dx; where=1; }
	if (min>fixed1->dy) { min=fixed1->dy; where=1; }
	if (min>fixed1->dz) { min=fixed1->dz; where=1; }
	if (where==2) { // must interpolate fixed image
	  fixed = LinearInterp(fixed1,min,min,min);
	  DestroyScene(&fixed1);
	}
	else { // must interpolate moving image
	  if (wxMessageBox(_T("Original volume will be interpolated to match the voxel dimensions of the reference volume! Continue?"), _T("Confirmation"), wxYES_NO, NULL)==wxNO) {
	    DestroyScene(&fixed1);
	    return;
	  }
	  Scene *tmp = LinearInterp(scn,min,min,min);
	  APP->SetDataVolume(tmp);
	  scn = tmp;
	}
      }
	  
      APP->Busy((char *)"Please wait, working...");
      APP->StatusMessage((char *)"Please wait - Computation in progress...");


      float T[4][4];
      float *best_theta;
      Register3(fixed,scn,T,&best_theta);  
      
      Scene *reg=CreateScene(scn->xsize,scn->ysize,scn->zsize);      
      transformScene(scn,T,reg);

      APP->StatusMessage((char *)"Done.");
      APP->Unbusy();

      /*
      char str[500];
      sprintf(str,"Registration Parameters: \nRx=%f\nRy=%f\nRz=%f\nTx=%f\nTy=%f\nTz=%f\nS=%f\n\n",best_theta[0],best_theta[1],best_theta[2],best_theta[3],best_theta[4],best_theta[5],best_theta[6]);
      sprintf(str,"%sTransform Matrix:\n%03.2f %03.2f %03.2f %03.2f \n%03.2f %03.2f %03.2f %03.2f \n%03.2f %03.2f %03.2f %03.2f \n%03.2f %03.2f %03.2f %03.2f \n",str,T[0][0],T[0][1],T[0][2],T[0][3],T[1][0],T[1][1],T[1][2],T[1][3],T[2][0],T[2][1],T[2][2],T[2][3],T[3][0],T[3][1],T[3][2],T[3][3]);
      wxString wxstr(str, wxConvUTF8);
      wxMessageBox(wxstr, _T("Registration results"), wxOK, APP->Window);
      */

      Scene *R,*G,*B;
      ColorReg(fixed,reg,&R,&G,&B);
      
      RigidRegDialog *dia = new RigidRegDialog(R,G,B);
      if (dia->ShowModal()==wxID_OK) {
	APP->SetDataVolume_NoDestroy(reg);
      }
      DestroyScene(&fixed);
      DestroyScene(&R);
      DestroyScene(&G);
      DestroyScene(&B);

      /*
      // Trasnforming the objects
      SegmObject *obj=NULL;
      int i;
      int n = APP->segmobjs->n;
      scn = APP->Data.orig;
      for (i=0; i < n; i++) {
	obj = (SegmObject *)GetArrayListElement(APP->segmobjs, i);
	printf("Transforming object %s...\n",obj->name);
	Scene *mask = CreateScene(scn->xsize,scn->ysize,scn->zsize);
	CopyBMap2SceneMask(mask,obj->mask);
	Scene *newmask = CreateScene(mask->xsize,mask->ysize,mask->zsize);    
	MytransformScene(mask,T,newmask);
	DestroyScene(&mask);
	BMap *newbmap;
	newbmap = SceneMask2BMap(newmask);
	DestroyScene(&newmask);
	BMapCopy(obj->mask,newbmap);
	BMapDestroy(newbmap);      
      }
      */

      APP->Refresh2DCanvas();
      APP->Refresh3DCanvas(true, 1.0);
    }
    APP->Refresh2DCanvas();
    APP->Refresh3DCanvas(true, 1.0);
    APP->DrawSegmObjects();
  }

  bool ModuleRigidReg :: Stop(){
    return true;
  }


} //end RigidReg namespace
