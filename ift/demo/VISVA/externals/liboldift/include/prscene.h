#ifndef _PRSCENE_H_
#define _PRSCENE_H_

#include "algebra.h"
#include "cimage.h"
#include "normal.h"
#include "analysis3.h"
#include "geometry.h"
#include "scene.h"
#include "adjacency.h"
#include "adjacency3.h"

// Voxel attributes
typedef struct _voxel_attr {
  ushort x,y,z; // scene coordinates  
  ushort normal_index; // index to normal table 
  uchar  object_index; // index to object table    
} Voxel_attr;


// Object attributes
typedef struct _obj_attr {
  uchar opac;     // object opacity 
  uchar visib;   // visibility code: 0-invisible 1-visible 
  uchar R, G, B; // object color    
} Obj_attr;

typedef struct _prscene {
  // List of voxels for visualization
  Voxel_attr *voxel; 
  int nvoxels; // number of voxels in the list
  int xsize,ysize,zsize; /* scene size */

  // lookup tables of diagonal size and constants that accelerate projection
  float *lkt_a1,   *lkt_a2;   
  float *lkt_A2b1, *lkt_C2b1; 
  float *lkt_B2b2, *lkt_D2b2; 
  float  E2, F2, b1_min, b2_min;
  // lookup tables that accelerate shading 
  Vector   *normal_table; // size = 65535 normal vectors 
  Obj_attr *object_table; // size = 255 objects (from 1)
  int nobjects; // number of objects in object_table
  int *depth_shading_table; // size = diagonal
  float d_min,d_max; // minimum and maximum projected depths

  // viewing parameters
  float R[4][4]; // rotation matrix 
  char paxis; // principal axis: x, y or z 
  Vector n; // normal to the view plane
  Point  p1, p2, p3, p4; // coordinates of the view plane

  // viewing buffers 
  int   *index_buffer;
  float *depth_buffer;
  int    i_min,i_max,j_min,j_max; // minimum and maximum coordinades of
			          // projected voxels
  
  // final image and scene diagonal 
  CImage *cimg; 
  int     diagonal;
} PRScene;

// create a PRScene
PRScene  *CreatePRScene(Scene *grey, Scene *label);
// render a PRScene, storing the final image
void RenderPRScene(PRScene *prs, float thetaX, float thetaY);
// destroy a PRScene
void      DestroyPRScene(PRScene **prs);
// update the PRScene, using new angles
void      UpdatePRSceneView(PRScene *prs, float thetaX, float thetaY);
void      SimplerUpdatePRSceneView(PRScene *prs, float thetaX, float thetaY);
// update the PRScene display options
void      UpdatePRSceneDisplay(PRScene *prs); // not implemented

// Auxiliary functions

// compute voxels in the objects' border and the size of the voxel list
Scene *PRSObjectBorder(Scene *label, int *nvoxels); 
// create voxel list with attributes
Voxel_attr *PRSCreateVoxelList(Scene *grey, Scene *label, int *nvoxels); 
// create object table
Obj_attr *PRSCreateObjectTable();
// project voxel list
void   PRSProject (PRScene *prs); 
void   SimplerPRSProject (PRScene *prs); 
// compute shading 
void   PRSShading (PRScene *prs); 




#endif







