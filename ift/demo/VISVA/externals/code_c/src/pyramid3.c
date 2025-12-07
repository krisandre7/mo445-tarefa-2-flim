
#include "pyramid3.h"


Pyramid3 *CreatePyramid3(int nlayers){
  void (*clean)();
  Pyramid3 *pyr;

  clean = DestroyScene;
  pyr = (Pyramid3 *) calloc(1,sizeof(Pyramid3));
  if(pyr == NULL)
    Error(MSG1,"CreatePyramid3");

  pyr->L = CreateArrayList(nlayers);
  SetArrayListCleanFunc(pyr->L,
			clean);
  pyr->nlayers = nlayers;
  
  return pyr;
}

void      DestroyPyramid3(Pyramid3 **pyr){
  Pyramid3 *aux;

  aux = *pyr;
  if(aux != NULL){
    DestroyArrayList(&aux->L);
    free(aux);
    *pyr = NULL;
  }
}


int   CompPyramid3MaxLayers(Scene *scn){
  int size,maxlayers;

  maxlayers = 0;
  size = MIN( MIN(scn->xsize, scn->ysize),
	      scn->zsize );
  while(size > 8){
    size >>= 1;
    maxlayers++;
  }

  return maxlayers;
}


Pyramid3 *GaussianPyramid3(Scene *scn,
			   int nlayers){
  Pyramid3 *pyr;
  Scene *aux,*prev;
  int maxlayers,i;

  maxlayers = CompPyramid3MaxLayers(scn);

  nlayers = MIN(nlayers,
		maxlayers);

  pyr = CreatePyramid3(nlayers);

  prev = CopyScene(scn);
  AddArrayListElement(pyr->L,
		      (void *)prev);
  for(i=1; i<nlayers; i++){
    aux  = FastOptGaussianBlur3(prev);
    prev = Subsampling3(aux);
    AddArrayListElement(pyr->L,
			(void *)prev);
    DestroyScene(&aux);
  }
  
  return pyr;
}

Pyramid3 *AsfOCRecPyramid3(Scene *scn,
			   int nlayers){
  Pyramid3 *pyr=NULL;

  return pyr;
}


Pyramid3 *ThresholdPyramid3(Pyramid3 *pyr, int lower, int higher){
  Pyramid3 *tpyr=NULL;
  Scene *scn=NULL,*bin=NULL;
  int i;

  tpyr = CreatePyramid3(pyr->nlayers);

  for(i=0; i<pyr->nlayers; i++){
    scn = GetPyramid3LayerRef(pyr, i);
    bin = Threshold3(scn, lower, higher);
    AddArrayListElement(tpyr->L, (void *)bin);
  }
  return tpyr;
}


Scene    *GetPyramid3LayerRef(Pyramid3 *pyr, 
			      int layer){
  return (Scene *)GetArrayListElement(pyr->L, 
				      layer);
}


void  WritePyramid3Layers(Pyramid3 *pyr,
			  char *filename){
  char base[512];
  char file[512];
  Scene *scn;
  int i;
  strcpy(base,filename);
  RemoveFileExtension(base);

  for(i=0; i<pyr->nlayers; i++){
    scn = GetPyramid3LayerRef(pyr, i);
    sprintf(file,"%s%02d.scn",base,i+1);
    WriteScene(scn,file);
  }
}


/*
Scene *ConvertLabelBetweenPyr3Layers(Pyramid3 *pyr,
				     Scene *label,
				     int from_layer,
				     int to_layer){
  Scene *scn,*flabel;
  int p,q,n,dlayer;
  float Ds;
  Voxel v;

  dlayer = (to_layer - from_layer);
  Ds = pow(2.0, dlayer);

  scn = GetPyramid3LayerRef(pyr, to_layer);
  flabel = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n = scn->xsize*scn->ysize*scn->zsize;

  for(p=0; p<n; p++){
    v.x = ROUND(Ds*(VoxelX(scn,p)-scn->xsize/2.) + label->xsize/2.);
    v.y = ROUND(Ds*(VoxelY(scn,p)-scn->ysize/2.) + label->ysize/2.);
    v.z = ROUND(Ds*(VoxelZ(scn,p)-scn->zsize/2.) + label->zsize/2.);
    if(ValidVoxel(label,v.x,v.y,v.z)){
      q = VoxelAddress(label,v.x,v.y,v.z);
      flabel->data[p] = label->data[q];
    }
    else
      flabel->data[p] = 0;
  }

  return flabel;
}
*/

Voxel VoxelCorrespondencePyr3(Pyramid3 *pyr,
			      Voxel v,
			      int from_layer,
			      int to_layer){
  Scene *fscn,*tscn;
  Voxel u;
  float Ds;

  if(from_layer==to_layer) return v;

  Ds = pow(2.0, (from_layer - to_layer));
  fscn = GetPyramid3LayerRef(pyr, from_layer);
  tscn = GetPyramid3LayerRef(pyr, to_layer);
  u.x = ROUND(Ds*(v.x-fscn->xsize/2)) + tscn->xsize/2;
  u.y = ROUND(Ds*(v.y-fscn->ysize/2)) + tscn->ysize/2;
  u.z = ROUND(Ds*(v.z-fscn->zsize/2)) + tscn->zsize/2;
  return u;
}


void AddFrame32PyramidLayer(Pyramid3 *pyr, int layer, int sz){
  Scene *new,*scn;
  int Imin;

  scn  = GetPyramid3LayerRef(pyr, layer);
  Imin = MinimumValue3(scn);
  new  = AddFrame3(scn, sz, Imin);
  pyr->L->array[layer] = (void *)new;
  DestroyScene(&scn);
}


