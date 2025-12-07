
#ifndef _MODULECSF_H_
#define _MODULECSF_H_

#include "startnewmodule.h"
extern "C" {
#include "csf.h"
}

namespace CSF{

  class ModuleCSF : public SegmentationModule{
  public:
    ModuleCSF();
    ~ModuleCSF();
    void Start();
    bool Stop();
    void Finish();
    void Run();
    bool Automatic();
    char obj_name[1024];

  protected:
    Scene   *brain;
  };
} //end CSF namespace


#endif

