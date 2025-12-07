
#ifndef _LIVEWIREOIFT_H_
#define _LIVEWIREOIFT_H_

extern "C" {
#include "oldift.h"
#include "shared.h"
#include "priorityqueue.h"
}

namespace OIFT{

  int *ConfirmedPathArray(int *path, int src);

  int *Path2Array(Image *pred, int init, int dst);
  
  int *path_by_iftLiveWire(Image *cost, Image *pred,
			   Image *arcw, AdjRel *A, int Wmax,
			   PriorityQueue **pQ,
			   int init, int src, int dst);

  void path_clear_current(Image *cost, Image *pred,
			  PriorityQueue **pQ,
			  int src);

} //end OIFT namespace

#endif

