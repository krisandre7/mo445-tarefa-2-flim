
#include "module.h"

void freeModule(void **mem){
  if(*mem!=NULL){
    Module *mod = (Module *)*mem;
    delete mod;
    *mem=NULL;
  }
}

Module :: Module(){
  this->version = new SoftwareVersion(0,0,0);
  this->name[0] = '\0';
  this->author[0] = '\0';
}

Module :: Module(  SoftwareVersion& ver,
		 char *name,
		 char *author){
  this->version = new SoftwareVersion(ver);
  strcpy(this->name, name);
  strcpy(this->author, author);
}


Module :: Module(char *name,
		 char *author){
  this->version = new SoftwareVersion(0,0,0);
  strcpy(this->name, name);
  strcpy(this->author, author);
}


Module :: ~Module(){
  delete this->version;
}


void Module :: Start(){
}

bool Module :: Stop(){
  return true;
}

bool Module :: Automatic(){
  return false;
}


void Module :: SetVersion(  SoftwareVersion& ver){
  delete this->version;
  this->version = new SoftwareVersion(ver);
}

void Module :: SetName(char *name){
  strcpy(this->name, name);
}

void Module :: SetAuthor(char *author){
  strcpy(this->author, author);
}

SoftwareVersion * Module :: GetVersion(){
  return this->version;
}


void Module :: GetName(char *name){
  strcpy(name, this->name);
}


void Module :: GetAuthor(char *author){
  strcpy(author, this->author);
}


//Module::ModuleType Module :: GetType(){
//}


//--------------------------------

PreProcModule :: PreProcModule(){
}

PreProcModule :: ~PreProcModule(){
}


Module::ModuleType PreProcModule :: GetType(){
  return Module::PREPROC;
}

//--------------------------------


RegistrationModule :: RegistrationModule(){
}

RegistrationModule :: ~RegistrationModule(){
}


Module::ModuleType RegistrationModule :: GetType(){
  return Module::REGISTRATION;
}


//--------------------------------

SegmentationModule :: SegmentationModule(){
}

SegmentationModule :: ~SegmentationModule(){
}


Module::ModuleType SegmentationModule :: GetType(){
  return Module::SEGMENTATION;
}

//--------------------------------

AnalysisModule :: AnalysisModule(){
}

AnalysisModule :: ~AnalysisModule(){
}


Module::ModuleType AnalysisModule :: GetType(){
  return Module::ANALYSIS;
}

//--------------------------------

RefreshHandler :: RefreshHandler(){
}


RefreshHandler :: ~RefreshHandler(){
}


void RefreshHandler :: OnRefresh2D(char axis){
}

//--------------------------------

View2DModule :: View2DModule(){
}

View2DModule :: ~View2DModule(){
}

Module::ModuleType View2DModule :: GetType(){
  return Module::VIEW2D;
}

void View2DModule :: SetCursor(  wxCursor *cursor,
			       char axis){
}

//--------------------------------

View3DModule :: View3DModule(){
}

View3DModule :: ~View3DModule(){
}

Module::ModuleType View3DModule :: GetType(){
  return Module::VIEW3D;
}

