
#ifndef _LIVEWIREDRIFT_H_
#define _LIVEWIREDRIFT_H_

extern "C" {
#include "oldift.h"
#include "shared.h"
#include "priorityqueue.h"
}

namespace DRIFT{

  int *ConfirmedPathArray(int *path, int src);

  int *Path2Array(Image *pred, int init, int dst);
  
  int *path_by_iftLiveWire(Image *cost, Image *pred,
			   Image *arcw, AdjRel *A, int Wmax,
			   PriorityQueue **pQ,
			   int init, int src, int dst);

  void path_clear_current(Image *cost, Image *pred,
			  PriorityQueue **pQ,
			  int src);

} //end DRIFT namespace

#endif

