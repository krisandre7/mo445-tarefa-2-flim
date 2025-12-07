#ifndef _ANALYSIS3_H_
#define _ANALYSIS3_H_


#include "annscn.h"
#include "genift3.h"
#include "adjacency3.h"
#include "radiometric3.h"

void   iftDistTrans3(AnnScn *ascn, AdjRel3 *A, char side);
void   iftGenDistTrans3(AnnScn *ascn, AdjRel3 *A, PathCost3 Pcost);
Scene *DistTrans3(Scene *bin, AdjRel3 *A, char side, char sign);
Scene *TDistTrans3(Scene *bin, AdjRel3 *A, char side, int limit, char sign);
Scene *Label2Border(Scene *lscn, AdjRel3 *A, char side, int limit);
Scene *Area3(Scene *bin);

#endif  







