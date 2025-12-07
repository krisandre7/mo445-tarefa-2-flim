
#ifndef _MODULECLOUDSBRAIN_H_
#define _MODULECLOUDSBRAIN_H_

#include "startnewmodule.h"

#include "cloudsbrain.h"

namespace CloudsBrain{

  class ModuleCloudsBrain : public SegmentationModule{
  public:
    ModuleCloudsBrain();
    ~ModuleCloudsBrain();
    void Start();
    bool Stop();

  protected:
  };

} //end CloudsBrain namespace

#endif

