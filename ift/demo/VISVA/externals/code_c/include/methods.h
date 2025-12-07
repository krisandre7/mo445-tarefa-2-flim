
#ifndef _METHODS_H_
#define _METHODS_H_

#include "oldift.h"

#include "priorityqueue.h"
#include "markerlist.h"

void   RunIFT(Scene *grad, Scene *cost, 
	      Scene *pred, Scene *mark, 
	      Set *seedSet);

void   RunDIFT(Scene *grad, Scene *cost, 
	       Scene *pred, Scene *mark, 
	       Set **seedSet, Set **delSet);

Set   *ForestRemoval(AdjRel3 *A, Scene *cost, 
		     Scene *pred, Scene *mark, 
		     Set **seedSet, Set **delSet);


//Returns the MST as a predecessor map (i.e., Scene *pred).
Scene *RunMST(Scene *grad, int root);
Scene *RunConstrainedMST(Scene *bin, Scene *grad);


Scene *GetIFTCores(Scene *bin, Scene *grad, 
		   Scene *energy, AdjRel3 *A);
void RefineIFTCores(Scene *cores, Scene *grad, 
		    Scene *energy, AdjRel3 *A);
void RemoveRedundantIFTCores(Scene *cores, Scene *grad, 
			     Scene *energy, Set *seedSet,
			     AdjRel3 *A);
void SelectSeedsInIFTCores(Scene *cores, Set **seedSet);


//Resumes/recovers a given segmentation (label) from scratch.
//Recovery is made by recomputing a seed set.
//
//--->Seeds are taken as the OPF prototypes.
void   ResumeFromScratchDIFT_Prototype(Scene *grad, Scene *label,
				       Scene *cost, Scene *pred, 
				       Scene *mark, Set **seedSet);
void   ResumeFromScratchDIFT_MinSeeds(Scene *grad, Scene *label,
				      Scene *cost, Scene *pred, 
				      Scene *mark, Set **seedSet);


//Resumes/recovers a given segmentation (label) with seeds.
//Recovery is made by computing an OPF that leads to the same results.
//Note: The gradient must be the same.
void   ResumeIFT(Scene *label, Scene *grad,
		 Scene *cost, Scene *pred,
		 Scene *mark, Set *seedSet);

#endif

