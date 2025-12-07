
#ifndef _MRI_H_
#define _MRI_H_

#include "oldift.h"
#include "shared.h"
#include "scene_addons.h"
#include "mmorphology.h"
#include "preproc.h"
#include "radiometric_addons.h"
#include "adjacency_addons.h"
#include "selection3.h"

typedef struct _MRI_Info {
  Voxel COG; //CentreOfGravity.
  int Tcsf;  //CSF estimation.
  int Ecsf;  //CSF error.
  int Tgm;   //GM estimation.
  int Twm;   //WM estimation.
  int Tup;   //UpperCut.
} MRI_Info;


#include "gradient.h"


//----------------------------------------------

MRI_Info EstimateMRI_Information(Scene *scn);

int    MRI_UpperCut(Scene *scn);
int    MRI_LowerCut(Scene *scn);
int    MRI_LowerBoundCSF(Scene *scn);
Voxel  MRI_CentreOfGravity(Scene *scn, int Tup);
int    MRI_EstimateCSF(Scene *scn, Voxel COG, int *Ecsf);
void   MRI_EstimateGM_WM(Scene *scn, int Tcsf, int Ecsf, int Tup,
			 int *Tgm, int *Twm);
Scene *MRI_InternalSeeds(Scene *scn, int Tcsf, int Ecsf,
			 int Tgm, int Tup);

void   MRI_SuppressSkull(Scene *arcw, Scene *scn, MRI_Info info);
void   MRI_SuppressOpticNerve(Scene *arcw, Scene *scn, MRI_Info info);
void   MRI_SuppressNonBrainBorders(Scene *arcw, 
				   Scene *scn,
				   ScnGradient *grad,
				   MRI_Info info);

void    ComputeMRISceneMBB(Scene *scn,
			   Voxel *Vm,
			   Voxel *VM);

Scene *MRI_FillObjInterfaces(Scene *label,
			     Scene *scn,
			     float r,
			     MRI_Info info);

//-------------------------------------------------

#endif

