
#ifndef _MODULEBRAIN_H_
#define _MODULEBRAIN_H_

#include "startnewmodule.h"

namespace Brain{

  class ModuleBrain : public SegmentationModule{
  public:
    ModuleBrain();
    ~ModuleBrain();
    void Start();
    bool Stop();
    void Run();
    bool Automatic();

  protected:
  };
} //end Brain namespace


#endif

