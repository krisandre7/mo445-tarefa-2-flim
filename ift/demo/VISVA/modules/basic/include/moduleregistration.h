#ifndef _MODULEREGISTRATION_H_
#define _MODULEREGISTRATION_H_

#include "module.h"
#include "modulepanel.h"

#include "visva.h"
extern BIAClass *APP;

class ModuleRegistration {
public:
  ModuleRegistration();
  ~ModuleRegistration();

  void Register(Module *m);
  int  GetNumberModules(Module::ModuleType type);
  Module *GetModule(Module::ModuleType type, 
		    int index);
  void ShowModuleWindow(Module::ModuleType type);
  bool StopLastActiveProcessingModule();

  //**** Dependency functions: ***********
  bool CheckDependency(char *name, Module::ModuleType type);
  bool SolveDependency(char *name, Module::ModuleType type);
  void SetDependencyStatus(char *name,
			   Module::ModuleType type,
			   bool solved);
  void RegisterDependencyHandler(char *name, Module *mod);
  void ClearDependencies();

protected:
  void DependencyFullName(char *dest, char *name,
			  Module::ModuleType type);
  ArrayList *M[Module::LAST_TYPE];
  Module    *prev;

  //**** Dependency data: ***********
  char         *dependency[512];
  Module       *dep_handler[512];
  bool          dep_solved[512];
  int           ndependencies;
};

#endif

