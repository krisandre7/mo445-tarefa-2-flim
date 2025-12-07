#ifndef _CSF_H_
#define _CSF_H_

#include "oldift.h"

void Force2ClustersCSF(Subgraph *sg, Scene *scn, Scene *mask, float T );
int  CSFThreshold(Scene *scn, Scene *mask);

#endif
