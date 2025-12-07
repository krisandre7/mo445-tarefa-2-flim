#ifndef _GENIFT3_H_
#define _GENIFT3_H_

#include "annscn.h"
#include "adjacency3.h"


/* path-cost functions */

typedef int (*PathCost3)(AnnScn *ascn, int s, int t);

int FEucl3(AnnScn *ascn, int s, int t); /* For 3D Euclidean distance
                                          transform and related
                                          operators */

int FEuclIn3(AnnScn *ascn, int s, int t); /* For 3D Euclidean distance 
					   inside de object */

int FEuclOut3(AnnScn *ascn, int s, int t); /* For 3D Euclidean distance 
					   outside de object */

/* Generic IFT Algorithm using a balanced heap: 
   ascn is an annotated scene, 
   A is an adjacency relation,
   Pcost is a smooth path-cost function.    
   maxcost is the maximum allowed value of Pcost. 
*/

void IFT3(AnnScn *ascn, AdjRel3 *A, PathCost3 Pcost, int maxinc); 


#endif
