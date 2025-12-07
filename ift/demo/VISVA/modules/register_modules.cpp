
#include "register_modules.h"

//Include your modules headers here.

#include "modulesliceview.h"
#include "modulesurface.h"
#include "modulerayleaping.h"
#include "modulecurvilinear.h"
#include "moduleinteractive.h"
#include "moduleroi.h"
#include "moduleenhancement.h"
#include "modulemanual.h"
#include "modulecombine.h"
#include "moduleventr.h"
#include "modulecsf.h"
#include "modulebrain.h"
#include "modulebasicstats.h"
#include "modulesymmetry.h"
#include "moduleintprofile.h"
#include "moduleinterpolation.h"
#include "modulemsp.h"
#include "modulerigidreg.h"
#include "moduleselfreg.h"
#include "moduleinhomogeneity.h"
#include "modulethreshold.h"
#include "modulebraincluster.h"
#include "modulesetorientation.h"
#include "modulecloudsbrain.h"
#include "modsegexternal.h"
#include "moduledrift.h"
#include "moduleoift.h"

void app_register_modules(ModuleRegistration *modManager){
  Module *mod = NULL;
  
  // Segmentation --------------------------------------

  mod = new Manual::ModuleManual();
  modManager->Register(mod);

  mod = new Combine::ModuleCombine();
  modManager->Register(mod);

  mod = new Interactive::ModuleInteractive();
  modManager->Register(mod);

  mod = new DRIFT::ModuleDRIFT();
  modManager->Register(mod);

  mod = new OIFT::ModuleOIFT();
  modManager->Register(mod);

  mod = new ModThreshold::ModuleThreshold();
  modManager->Register(mod);

  //Skull Stripping module
  mod = new Brain::ModuleBrain();
  modManager->RegisterDependencyHandler((char *)"Brain", mod);
  modManager->RegisterDependencyHandler((char *)"Brain Envelope", mod);
  modManager->Register(mod);

  mod = new CloudsBrain::ModuleCloudsBrain();
  modManager->Register(mod);

  mod = new BrainCluster::ModuleBrainCluster();
  modManager->Register(mod);

  mod = new SegExternal::ModSegExternal((char *)"Brain: BET", (char *)"---");
  modManager->Register(mod);

  mod = new CSF::ModuleCSF();
  modManager->RegisterDependencyHandler((char *)"CSF", mod);

  mod = new Ventr::ModuleVentr();
  modManager->Register(mod);

  // Viewing --------------------------------------

  mod = new SliceView::ModuleSliceView();
  modManager->Register(mod);

  mod = new Surface::ModuleSurface();
  modManager->Register(mod);

  mod = new Rayleaping::ModuleRayleaping();
  modManager->Register(mod);

  mod = new Curvilinear::ModuleCurvilinear();
  modManager->Register(mod);

  // Analisys --------------------------------------

  mod = new BasicStats::ModuleBasicStats();
  modManager->Register(mod);

  mod = new IntensityProfile::ModuleIntensityProfile();
  modManager->Register(mod);

  if (getenv("BIA_EXPERIMENTAL")) {
    mod = new Symmetry::ModuleSymmetry();
    modManager->Register(mod);
  }



  // Preprocessing --------------------------------------

  mod = new Interpolation::ModuleInterpolation();
  modManager->Register(mod);

  mod = new Setorientation::ModuleSetorientation();
  modManager->Register(mod);

  mod = new MSP::ModuleMSP();
  modManager->Register(mod);

  mod = new Inhomogeneity::ModuleInhomogeneity();
  modManager->Register(mod);

  mod = new ModROI::ModuleROI();
  modManager->Register(mod);

  mod = new Enhancement::ModuleEnhancement();
  modManager->Register(mod);

  // Registration --------------------------------------

  mod = new RigidReg::ModuleRigidReg();
  modManager->Register(mod);

  mod = new SelfReg::ModuleSelfReg();
  modManager->Register(mod);

}


