
#ifndef _DYNAMICID_H_
#define _DYNAMICID_H_

extern "C" {
#include "oldift.h"
}

class DynamicID {
public:
  DynamicID(int base_id);
  ~DynamicID();
  int  AllocID();
  void FreeID(int id);
private:
  int base_id;
  int id;
  Set *R;
};

#endif

