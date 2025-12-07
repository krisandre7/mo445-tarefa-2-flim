
#ifndef _OBJGROUP_H_
#define _OBJGROUP_H_

#include "oldift.h"
#include "compressed.h"
#include "filelist.h"
#include "shared.h"
#include "evaluation3.h"


/* Find a maximal clique containing element 'i' in a graph 
with 'n' vertices defined by an adjacency matrix A.*/
int *MaximalClique(int **A, int n, int i);

/* Find groups of shapes with intra-similarities above a threshold
   within (0,1) */
void GroupsByMaximalCliques(RealMatrix *arcweights,
			    FileList *L, float simil,
			    char *outputdir);

RealMatrix *ShapeSimilarityMatrix3(FileList *L);
RealMatrix *MObjSimilarityMatrix3(FileList *L);

#endif

