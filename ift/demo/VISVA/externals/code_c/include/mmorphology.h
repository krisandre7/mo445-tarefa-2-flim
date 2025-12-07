
#ifndef _MMORPHOLOGY_H_
#define _MMORPHOLOGY_H_

#include "oldift.h"
#include "shared.h"
#include "scene_addons.h"

//The radius value should be given in millimeters.
Scene *ErodeBinScnRadial(Scene *bin, Voxel C, float r);

//The radius value should be given in millimeters.
Scene *ErodeBinScn(Scene *bin, Set **seed, float radius);
Scene *DilateBinScn(Scene *bin, Set **seed, float radius);
Scene *CloseBinScn(Scene *bin, float radius);
Scene *OpenBinScn(Scene *bin, float radius);

#endif



