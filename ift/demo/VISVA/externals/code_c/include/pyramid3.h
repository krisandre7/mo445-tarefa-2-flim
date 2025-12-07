
#ifndef _PYRAMID3_H_
#define _PYRAMID3_H_

#include "oldift.h"
#include "shared.h"
#include "preproc.h"
#include "arraylist.h"
#include "filelist.h"

typedef struct _Pyramid3 {
  ArrayList *L;
  int nlayers;
} Pyramid3;

Pyramid3 *CreatePyramid3(int nlayers);
void      DestroyPyramid3(Pyramid3 **pyr);

int       CompPyramid3MaxLayers(Scene *scn);

Pyramid3 *GaussianPyramid3(Scene *scn,
			   int nlayers);
Pyramid3 *AsfOCRecPyramid3(Scene *scn,
			   int nlayers);
Pyramid3 *ThresholdPyramid3(Pyramid3 *pyr, int lower, int higher);

Scene    *GetPyramid3LayerRef(Pyramid3 *pyr, 
			      int layer);

void      WritePyramid3Layers(Pyramid3 *pyr,
			      char *filename);

/*
Scene    *ConvertLabelBetweenPyr3Layers(Pyramid3 *pyr,
					Scene *label,
					int from_layer,
					int to_layer);
*/

Voxel VoxelCorrespondencePyr3(Pyramid3 *pyr,
			      Voxel v,
			      int from_layer,
			      int to_layer);


void  AddFrame32PyramidLayer(Pyramid3 *pyr, int layer, int sz);


#endif

