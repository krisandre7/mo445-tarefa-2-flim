
#include "moduleregistration.h"


ModuleRegistration :: ModuleRegistration(){
  int i;

  this->prev = NULL;
  this->ndependencies = 0;
  for(i=0; i<Module::LAST_TYPE; i++){
    this->M[i] = CreateArrayList(20);
    SetArrayListCleanFunc(this->M[i], freeModule);
  }
}


ModuleRegistration :: ~ModuleRegistration(){
  int i;
  
  for(i=0; i<Module::LAST_TYPE; i++)
    DestroyArrayList(&this->M[i]);
}


void ModuleRegistration :: Register(Module *m){
  AddArrayListElement(this->M[m->GetType()],
		      (void *)m);
}

int ModuleRegistration :: GetNumberModules(Module::ModuleType type){
  return (M[type]->n);
}


Module* ModuleRegistration :: GetModule(Module::ModuleType type, 
					int index){
  return (Module *)GetArrayListElement(M[type], index);
}


void ModuleRegistration :: ShowModuleWindow(Module::ModuleType type){
  char title[512],msg[512];
  int cod,id,j;

  if( !StopLastActiveProcessingModule() ) return;

  switch(type){
  case Module::PREPROC:
    strcpy(title, "Transformations"); //Preprocessing");
    break;
  case Module::SEGMENTATION:
    strcpy(title, "Segmentation");
    break;
  case Module::ANALYSIS:
    strcpy(title, "Analysis");
    break;
  case Module::REGISTRATION:
    strcpy(title, "Registration");
    break;
  case Module::VIEW2D:
    strcpy(title, "2D Visualization");
    break;
  case Module::VIEW3D:
    strcpy(title, "3D Visualization");
    break;
  case Module::LAST_TYPE:
    return;
  }

  if(M[type]->n == 0){
    sprintf(msg,"No %s modules available.",title);
    wxString wxstr(msg, wxConvUTF8);
    wxMessageBox(wxstr, _T("Warning"),
		 wxOK | wxICON_EXCLAMATION, APP->Window);
    return;
  }


  id = APP->idManager->AllocID();
  BaseDialog dialog(APP->Window, title);
  ModulePanel panel((wxWindow *)&dialog, id);

  for(j=0; j<M[type]->n; j++)
    panel.AddModule((Module *)GetArrayListElement(M[type], j));

  dialog.AddPanel(&panel);
  cod = dialog.ShowModal();

  if(cod==wxID_OK){
    Module *m = panel.GetSelection();
    
    if(type==Module::VIEW2D){
      View2DModule *m2d = (View2DModule *)m;
      APP->Set2DViewModule(m2d);
    }
    else if(type==Module::VIEW3D){
      View3DModule *m3d = (View3DModule *)m;
      APP->Set3DViewModule(m3d);
    }
    else{
      this->prev = m;
    }
    m->Start();
  }
  APP->idManager->FreeID(id);
}


bool ModuleRegistration :: StopLastActiveProcessingModule(){
  bool ret;
  if(this->prev!=NULL){
    ret = (this->prev)->Stop();
    if(ret) this->prev = NULL;
    return ret;
  }
  return true;
}


//**** Dependency functions: ***********

void ModuleRegistration::DependencyFullName(char *dest,
					    char *name,
					    Module::ModuleType type){
  if(type==Module::PREPROC)
    sprintf(dest,"PRE_%s",name);
  else if(type==Module::SEGMENTATION)
    sprintf(dest,"SEG_%s",name);
  else if(type==Module::ANALYSIS)
    sprintf(dest,"ANA_%s",name);
  else if(type==Module::REGISTRATION)
    sprintf(dest,"REG_%s",name);
  else if(type==Module::VIEW2D)
    sprintf(dest,"2D_%s",name);
  else
    sprintf(dest,"3D_%s",name);
}


bool ModuleRegistration :: CheckDependency(char *name, 
					   Module::ModuleType type){
  char aux[1024];
  bool check = false;
  int i;

  DependencyFullName(aux, name, type);

  for(i=0; i<this->ndependencies; i++)
    if(strcmp(this->dependency[i],aux)==0)
      if(this->dep_solved[i]==true)
	check = true;

  return check;
}


bool ModuleRegistration :: SolveDependency(char *name, 
					   Module::ModuleType type){
  char aux[1024];
  bool solve = false;
  int i;
  Module *mod;

  DependencyFullName(aux, name, type);

  for(i=0; i<this->ndependencies; i++){
    if(strcmp(this->dependency[i],aux)==0){
      mod = this->dep_handler[i];
      if(mod!=NULL){
	solve = mod->Automatic();
	this->dep_solved[i] = solve;
	break;
      }
    }
  }
  return solve;
}


void ModuleRegistration::SetDependencyStatus(char *name,
					     Module::ModuleType type,
					     bool solved){
  char aux[1024];
  int i;

  DependencyFullName(aux, name, type);

  for(i=0; i<this->ndependencies; i++){
    if(strcmp(this->dependency[i],aux)==0){
      this->dep_solved[i] = solved;
      break;
    }
  }
}


void ModuleRegistration :: RegisterDependencyHandler(char *name, 
						     Module *mod){
  char aux[1024];
  Module::ModuleType type = mod->GetType();

  DependencyFullName(aux, name, type);

  this->dependency[ndependencies] = strdup(aux);
  this->dep_handler[ndependencies] = mod;
  this->dep_solved[ndependencies] = false;
  this->ndependencies++;
}



void ModuleRegistration :: ClearDependencies(){
  int i;

  for(i=0; i<this->ndependencies; i++)
    this->dep_solved[i] = false;
}


