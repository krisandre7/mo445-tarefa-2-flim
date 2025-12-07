
#ifndef _MODSEGEXTERNAL_H_
#define _MODSEGEXTERNAL_H_

#include "startnewmodule.h"

namespace SegExternal{

  class ModSegExternal : public SegmentationModule{
  public:
    ModSegExternal(char *name, char *author);
    ~ModSegExternal();
    void Start();
    bool Stop();

  protected:
  };
} //end SegExternal namespace


#endif

