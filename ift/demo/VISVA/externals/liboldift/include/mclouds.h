#ifndef _MCLOUDS_H_
#define _MCLOUDS_H_

#include "adjacency.h"

typedef struct _mclouds { /* feature map */
  AdjRel **A; /* a vector of adjacency relations for each cloud */  
  int n;      /* the number of clouds */
} MClouds;

MClouds *CreateMClouds(int n);
void     DestroyMClouds(MClouds **mclouds);

/* r1 is the radius of the pupil (inner circle), r2 and r3 are the
   inner and outer radius of the external ring. The center of the ring
  is the reference point. */
MClouds *RoundEye(float r1, float r2, float r3);  

/* Two balls with radius r1 and r2, respectively, where the center of
   the first ball is the reference point and the second ball is
   (dx,dy) from the first one. */

MClouds *TwoBalls(float r1, float r2, int dx, int dy); 

Image *DrawMClouds(MClouds *mclouds); /* return a labeled image where
					each cloud has a label 1, 2, n. */

#endif
