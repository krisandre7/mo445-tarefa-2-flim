//Calculates and plots the intensity values along a line segment.

#ifndef _MODULEINTPROFILE_H_
#define _MODULEINTPROFILE_H_

#include "startnewmodule.h"

namespace IntensityProfile{

  class ModuleIntensityProfile : public AnalysisModule{
  public:
    ModuleIntensityProfile();
    ~ModuleIntensityProfile();
    void Start();
    bool Stop();

  protected:
    InteractionHandler *handler;
  };
}

#endif

