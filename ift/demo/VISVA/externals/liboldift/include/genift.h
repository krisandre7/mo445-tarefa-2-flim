#ifndef _GENIFT_H_
#define _GENIFT_H_

#include "annimg.h"
#include "adjacency.h"


/* path-cost functions */

typedef int (*PathCost)(AnnImg *aimg, int p, int q);

int FPeak(AnnImg *aimg, int p, int q); /* For Watershed on the
					  gradient and Superior
					  Reconstructions */
int FSum(AnnImg *aimg, int p, int q);  /* For Edge Tracking on the gradient */
int FEucl(AnnImg *aimg, int p, int q); /* For Euclidean distance
                                          transform and related
                                          operators */
int FMax(AnnImg *aimg, int p, int q); /*  For relative fuzzy connectedness */
int FCham(AnnImg *aimg, int p, int q); /* Chamfer 17x24 */
int FIni(AnnImg *aimg, int p, int q);  /* For regional minima */
int FLPeak(AnnImg *aimg, int p, int q); /* For Local Watershed on the
					  gradient and Local Superior
					  Reconstructions */

/* Generic IFT Algorithm using a balanced heap: 
   aimg is an annotated image, 
   A is an adjacency relation,
   Pcost is a smooth path-cost function.    
   maxinc is the maximum increment of Pcost.
   maxcost is the maximum allowed value of Pcost. 
*/

void IFT(AnnImg *aimg, AdjRel *A, PathCost Pcost, int maxinc);


#endif
