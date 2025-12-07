
#ifndef _BIA_IFT_H_
#define _BIA_IFT_H_

extern "C" {
#include "oldift.h"
#include "markerlist.h"
}

#include "bia_common.h"
#include "bia_pqueue16.h"
#include "bia_adjrel3.h"
#include "bia_scene16.h"
#include "bia_bmap.h"


void   RunIFT(bia::Scene16::Scene16 *grad, 
	      bia::Scene16::Scene16 *cost, 
	      Scene *pred, Scene *mark, 
	      Set *seedSet);

void   RunDIFT(bia::Scene16::Scene16 *grad, 
	       bia::Scene16::Scene16 *cost, 
	       Scene *pred, Scene *mark, 
	       Set **seedSet, Set **delSet);

Set   *ForestRemoval(bia::AdjRel3::AdjRel3 *A, 
		     bia::Scene16::Scene16 *cost, 
		     Scene *pred, Scene *mark, 
		     Set **seedSet, Set **delSet);


//Returns the MST as a predecessor map (i.e., Scene *pred).
Scene *RunMST(bia::Scene16::Scene16 *grad, int root);
Scene *RunConstrainedMST(Scene *bin, bia::Scene16::Scene16 *grad);


Scene *GetIFTCores(Scene *bin, 
		   bia::Scene16::Scene16 *grad, 
		   bia::Scene16::Scene16 *energy, 
		   bia::AdjRel3::AdjRel3 *A);
/*
void RefineIFTCores(Scene *cores, 
		    bia::Scene16::Scene16 *grad, 
		    bia::Scene16::Scene16 *energy, 
		    bia::AdjRel3::AdjRel3 *A);
*/
void RemoveRedundantIFTCores(Scene *cores, 
			     bia::Scene16::Scene16 *grad, 
			     bia::Scene16::Scene16 *energy, 
			     Set *seedSet,
			     bia::AdjRel3::AdjRel3 *A,
			     float nvol);
void SelectSeedsInIFTCores(Scene *scn, bia::Scene16::Scene16 *grad,
			   Scene *cores, Set **seedSet);
void SelectBkgSeedsInUniformGrid(Scene *label,
				 bia::Scene16::Scene16 *grad, 
				 Scene *mark,
				 Set **seedSet,
				 int ndiv);


//Resumes/recovers a given segmentation (label) from scratch.
//Recovery is made by recomputing a seed set.
//
//--->Seeds are taken as the OPF prototypes.
void   ResumeFromScratchDIFT_Prototype(bia::Scene16::Scene16 *grad, 
				       Scene *label,
				       bia::Scene16::Scene16 *cost, 
				       Scene *pred, 
				       Scene *mark, Set **seedSet);
void   ResumeFromScratchDIFT_MinSeeds(Scene *scn,
				      bia::Scene16::Scene16 *grad, 
				      Scene *label,
				      bia::Scene16::Scene16 *cost, 
				      Scene *pred, 
				      Scene *mark, Set **seedSet);


//Resumes/recovers a given segmentation (label) with seeds.
//Recovery is made by computing an OPF that leads to the same results.
//Note: The gradient must be the same.
void   ResumeIFT(Scene *label, 
		 bia::Scene16::Scene16 *grad,
		 bia::Scene16::Scene16 *cost, 
		 Scene *pred,
		 Scene *mark, Set *seedSet);

#endif

