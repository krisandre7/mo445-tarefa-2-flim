
#ifndef _MODULESYMMETRY_H_
#define _MODULESYMMETRY_H_

#include "startnewmodule.h"

namespace Symmetry{

  class ModuleSymmetry : public AnalysisModule{
  public:
    ModuleSymmetry();
    ~ModuleSymmetry();
    void Start();

    Scene* SymmetryDiffBinScene(Scene *scn);
    void AddMaskToSegmObj(Scene *mask, char *obj_name, int obj_color);

  protected:


  };
}

#endif

