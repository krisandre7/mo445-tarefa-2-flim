#include "ift.h"

/*---------- Annotated image ---------------------*/

typedef struct ift_annimg {
  iftImage *img;     /* input image */
  iftImage *pred;    /* predecessor image */
  iftImage *root;    /* root image */
  iftImage *cost;    /* path cost image */
  iftImage *label;   /* label image */
} iftAnnImg;

iftAnnImg *iftCreateAnnImg(iftImage     *img);
void       iftDestroyAnnImg(iftAnnImg **aimg);

/*---------- Path-value functions ----------------*/

typedef int (*iftPathCost)(iftAnnImg *aimg, iftImage *handicap, int p, int q);

int iftFsuprec(iftAnnImg *aimg, iftImage *handicap, int p, int q); /* For watershed transform and superior reconstructions */
int iftFmsf(iftAnnImg *aimg, iftImage *handicap, int p, int q);    /* For minimum spanning forest segmentation */
int iftFbtrack(iftAnnImg *aimg, iftImage *handicap, int p, int q); /* For boundary tracking */
int iftFedt(iftAnnImg *aimg, iftImage *handicap, int p, int q);    /* For Euclidean distance transform */
int iftFgeo(iftAnnImg *aimg, iftImage *handicap, int p, int q);    /* For geodesic dilations (chamfer 5x7) */

/*---------- 

  General ift algorithms with and without seeds: They guarantee a
  spanning forest even when the path-cost function does not satisfy
  the sufficient conditions. The lambda image assigns the initial
  labels to the seeds, while the handicap image assigns the initial
  cost values of trivial paths. In the case of no seeds, each tree is
  labeled by an integer from 1 to n.

--------------------------*/

iftAnnImg *iftGeneralAlgorithmWithSeeds(iftImage *img, iftAdjRel *A, iftImage *lambda, iftImage *handicap, iftPathCost Pcost);
iftAnnImg *iftGeneralAlgorithmWithNoSeeds(iftImage *img, iftAdjRel *A, iftImage *handicap, iftPathCost Pcost);

/* ------------ Other functions ----------------*/

iftImage *iftMyDrawBorders(iftAnnImg *aimg);
iftAnnImg *iftClassicWatershed(iftImage *img, iftAdjRel *A);
iftAnnImg *iftWatershedFromGrayMarker(iftImage *img, iftImage *marker, iftAdjRel *A);
iftAnnImg *iftSuperiorReconstruction(iftImage *img, iftImage *marker, iftAdjRel *A);
iftAnnImg *iftClosingByReconstruction(iftImage *img, iftAdjRel *A);
iftAnnImg *iftClosingOfBasins(iftImage *img);

/* ------------- Code of the function ----------------------------- */

iftAnnImg *iftCreateAnnImg(iftImage *img)
{
  iftAnnImg *aimg = (iftAnnImg *)calloc(1, sizeof(iftAnnImg));

  aimg->img     = iftCopyImage(img);
  aimg->pred    = iftCreateImage(img->xsize,img->ysize,img->zsize);
  aimg->root    = iftCreateImage(img->xsize,img->ysize,img->zsize);
  aimg->cost    = iftCreateImage(img->xsize,img->ysize,img->zsize);
  aimg->label   = iftCreateImage(img->xsize,img->ysize,img->zsize);

  return(aimg);
}

void  iftDestroyAnnImg(iftAnnImg **aimg)
{
  iftAnnImg *aux = *aimg;
    
  iftDestroyImage(&aux->img);
  iftDestroyImage(&aux->pred);
  iftDestroyImage(&aux->root);
  iftDestroyImage(&aux->cost);
  iftDestroyImage(&aux->label);
  free(aux);
  *aimg=NULL;
}

int iftFsuprec(iftAnnImg *aimg, iftImage *handicap, int p, int q){

  if (p==q) 
    return(handicap->val[p]); 
  else
    return(MAX(aimg->cost->val[p],aimg->img->val[q]));
}

int iftFmsf(iftAnnImg *aimg, iftImage *handicap, int p, int q)
{
  if (p==q) 
    return(handicap->val[p]); 
  else{
    if (aimg->img->val[p]!=aimg->img->val[q])
      return(INFINITY_INT);
    else
      return(0);
  }
}

iftAnnImg *iftGeneralAlgorithmWithSeeds(iftImage *img, iftAdjRel *A, iftImage *lambda, iftImage *handicap, iftPathCost Pcost)
{
  iftGQueue *Q=NULL;
  int        i,p,q,cost;
  iftVoxel   u,v;
  iftAnnImg *aimg=NULL;

  aimg = iftCreateAnnImg(img);
  Q    = iftCreateGQueue(QSIZE+1,img->n,aimg->cost->val); 

  for (p=0; p < img->n; p++) {
    aimg->pred->val[p]     = NIL;
    aimg->label->val[p]    = lambda->val[p]; 
    aimg->root->val[p]     = p; 
    aimg->cost->val[p]     = Pcost(aimg,handicap,p,p);
    iftInsertGQueue(&Q,p);
  }
  
  while(!iftEmptyGQueue(Q)) {
    p   = iftRemoveGQueue(Q);
    u   = iftGetVoxelCoord(img,p);

    for (i=1; i < A->n; i++){
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(img,v)){
	q = iftGetVoxelIndex(img,v);
	if (Q->L.elem[q].color != BLACK){ /* q belongs to Q */
	  cost = Pcost(aimg,handicap,p,q); 
	  if (cost < aimg->cost->val[q]){
	    iftRemoveGQueueElem(Q,q);
	    aimg->pred->val[q]  = p;
	    aimg->label->val[q] = aimg->label->val[p];
	    aimg->root->val[q]  = aimg->root->val[p];
	    aimg->cost->val[q]  = cost;
	    iftInsertGQueue(&Q,q);
	  }
	}
      }
    }
  }

  iftDestroyGQueue(&Q);
  return(aimg);
}

iftAnnImg *iftGeneralAlgorithmWithNoSeeds(iftImage *img, iftAdjRel *A, iftImage *handicap, iftPathCost Pcost)
{
  iftGQueue *Q=NULL;
  int        i,p,q,cost,label=1;
  iftVoxel   u,v;
  iftAnnImg *aimg=NULL;

  aimg = iftCreateAnnImg(img);
  Q    = iftCreateGQueue(QSIZE+1,img->n,aimg->cost->val); 

  for (p=0; p < img->n; p++) {
    aimg->pred->val[p]     = NIL;
    aimg->root->val[p]     = p; 
    aimg->cost->val[p]     = Pcost(aimg,handicap,p,p);
    iftInsertGQueue(&Q,p);
  }
  
  while(!iftEmptyGQueue(Q)) {
    p   = iftRemoveGQueue(Q);
    u   = iftGetVoxelCoord(img,p);

    if (aimg->pred->val[p]==NIL){ /* p is a root of the map */
      aimg->label->val[p]=label; label++; 
      aimg->cost->val[p]--; /* this guarantees a single root per
	   		       minimum of the cost map. For Fmsf, it
	   		       works as well. */
    }

    for (i=1; i < A->n; i++){
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(img,v)){
	q = iftGetVoxelIndex(img,v);
	if (Q->L.elem[q].color != BLACK){ /* q belongs to Q */
	  cost = Pcost(aimg,handicap,p,q); 
	  if (cost < aimg->cost->val[q]){
	    iftRemoveGQueueElem(Q,q);
	    aimg->pred->val[q]  = p;
	    aimg->label->val[q] = aimg->label->val[p];
	    aimg->root->val[q]  = aimg->root->val[p];
	    aimg->cost->val[q]  = cost;
	    iftInsertGQueue(&Q,q);
	  }
	}
      }
    }
  }

  iftDestroyGQueue(&Q);
  return(aimg);
}

iftAnnImg *iftClassicWatershed(iftImage *img, iftAdjRel *A)
{
  iftImage  *handicap = iftCreateImage(img->xsize,img->ysize,img->zsize);
  iftAnnImg *aimg  = NULL;
  int p; 

  for (p=0; p < img->n; p++) 
    handicap->val[p]=img->val[p]+1;

  aimg  = iftGeneralAlgorithmWithNoSeeds(img, A, handicap, iftFsuprec);

  iftDestroyImage(&handicap);

  return(aimg);
}

iftAnnImg *iftWatershedFromGrayMarker(iftImage *img, iftImage *marker, iftAdjRel *A)
{
  iftImage  *handicap = iftCreateImage(img->xsize,img->ysize,img->zsize);
  iftAnnImg *aimg  = NULL;
  int p; 

  for (p=0; p < img->n; p++) 
    handicap->val[p]=marker->val[p]+1;

  aimg  = iftGeneralAlgorithmWithNoSeeds(img, A, handicap, iftFsuprec);

  iftDestroyImage(&handicap);

  return(aimg);
}

iftAnnImg *iftSuperiorReconstruction(iftImage *img, iftImage *marker, iftAdjRel *A)
{
  iftImage  *handicap = iftCreateImage(img->xsize,img->ysize,img->zsize);
  iftAnnImg *aimg  = NULL;
  int p; 

  for (p=0; p < img->n; p++) 
    handicap->val[p]=marker->val[p]+1;

  aimg  = iftGeneralAlgorithmWithNoSeeds(img, A, handicap, iftFsuprec);

  iftDestroyImage(&handicap);

  return(aimg);
}

iftAnnImg *iftClosingByReconstruction(iftImage *img, iftAdjRel *A)
{
  iftImage  *marker   = iftClose(img,A);
  iftAdjRel *B=NULL;
  iftAnnImg *aimg;

  if (iftIs3DImage(img))
    B = iftSpheric(1.0);
  else
    B = iftCircular(1.0);

  aimg = iftSuperiorReconstruction(img,marker,B);

  iftDestroyImage(&marker);
  iftDestroyAdjRel(&B);

  return(aimg);
}

iftAnnImg *iftClosingOfBasins(iftImage *img)
{
  iftImage  *handicap = iftCreateImage(img->xsize,img->ysize,img->zsize);
  iftAnnImg *aimg  = NULL;
  iftAdjRel *A     = NULL;
  int p; 
  iftVoxel u;

  if (iftIs3DImage(img)){
    A = iftSpheric(1.0);
    for (p=0; p < img->n; p++){
      handicap->val[p]=INFINITY_INT;
      u = iftGetVoxelCoord(img,p);
      if ((u.x==0)||(u.y==0)||(u.z==0))
	handicap->val[p]=img->val[p]+1;
    }
  } else { /* 2D image */
    A = iftCircular(1.0);
    for (p=0; p < img->n; p++){
      handicap->val[p]=INFINITY_INT;
      u = iftGetVoxelCoord(img,p);
      if ((u.x==0)||(u.y==0)){
	handicap->val[p]=img->val[p]+1;
      }
    }
  }

  aimg  = iftGeneralAlgorithmWithNoSeeds(img, A, handicap, iftFsuprec);

  iftDestroyImage(&handicap);
  iftDestroyAdjRel(&A);

  return(aimg);
}

iftImage *iftMyDrawBorders(iftAnnImg *aimg)
{ 

  //  return(iftColorizeComp(aimg->label));

  iftColor   RGB, YCbCr;
  iftAdjRel *A    = iftCircular(1.0); /* defines border element */
  iftAdjRel *B    = iftCircular(0.0); /* thickness of the border */
  iftImage  *border = iftCopyImage(aimg->img);
 
  RGB.val[0] = 255;
  RGB.val[1] = 50;
  RGB.val[2] = 100;

  YCbCr = iftRGBtoYCbCr(RGB,255);

  iftDrawBorders(border, aimg->label, A, YCbCr, B);
  iftDestroyAdjRel(&B);
  B = iftCircular(3.0);
  iftDrawRoots(border,aimg->root, YCbCr, B);

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);

  return(border);
}

int main(int argc, char *argv[]) 
{
  iftJson   *json;
  char      *operator;
  iftAnnImg *aimg;

  if (argc!=2)
    iftError("Usage: gift <parameters.json>","main");

  json = iftReadJson(argv[1]);

  operator = iftGetJString(json, "operator");

  /* ------- Classic watershed (with one label per minimum) ------------ */
 
  if (strcmp(operator,"classic_watershed")==0){

    char     *input_image  = iftGetJString(json, "input_image");
    char     *output_image = iftGetJString(json, "output_image");
    iftImage *basins       = iftReadImageByExt(input_image);
    iftAdjRel *A           = NULL;

    if (iftIs3DImage(basins))
      A = iftSpheric(sqrtf(3.0));
    else
      A = iftCircular(sqrtf(2.0));
    
    aimg             = iftClassicWatershed(basins,A);

    if (iftIs3DImage(basins)){
      iftWriteImageByExt(aimg->label,output_image);
    } else {
      iftImage *border = iftMyDrawBorders(aimg);
      iftWriteImageByExt(border,output_image);
      iftDestroyImage(&border);
    }

    free(input_image);
    free(output_image);
    iftDestroyImage(&basins);
    iftDestroyAdjRel(&A);
  }

  /* ------- Watershed from Gray Marker (one root per minimum) ------------ */
 
  if (strcmp(operator,"watershed_from_gray_marker")==0){

    char     *input_image  = iftGetJString(json, "input_image");
    char     *output_image = iftGetJString(json, "output_image");
    iftImage *basins       = iftReadImageByExt(input_image);


    /* Filter criterion: by height, area, or volume */

    char     *criterion    = iftGetJString(json, "criterion"); 
    int       threshold    = iftGetJInt(json, "threshold");
    iftImage *marker       = NULL;
    iftAdjRel *A           = NULL;

    if (strcmp(criterion,"height")==0){
      marker = iftAddValue(basins,threshold);
    } else {
      if (strcmp(criterion,"area")==0){
	marker = iftFastAreaClose(basins,threshold);
      } else {
	if (strcmp(criterion,"volume")==0){
	  marker = iftVolumeClose(basins,threshold);
	}
      }
    }

    if (iftIs3DImage(basins))
      A = iftSpheric(sqrtf(3.0));
    else
      A = iftCircular(sqrtf(2.0));
    
    aimg             = iftWatershedFromGrayMarker(basins,marker,A);

    if (iftIs3DImage(basins)){
      iftWriteImageByExt(aimg->label,output_image);
    } else {
      iftDestroyImage(&aimg->img);
      aimg->img = iftCopyImage(aimg->cost);
      iftImage *border = iftMyDrawBorders(aimg);
      iftWriteImageByExt(border,output_image);
      iftDestroyImage(&border);
    }

    free(input_image);
    free(output_image);
    free(criterion);
    iftDestroyImage(&marker);
    iftDestroyImage(&basins);
    iftDestroyAdjRel(&A);
  }

  /* ------- Superior Reconstruction (preserves the values of the
             remaining regions) ------------ */
 
  if (strcmp(operator,"superior_reconstruction")==0){

    char     *input_image  = iftGetJString(json, "input_image");
    char     *output_image = iftGetJString(json, "output_image");
    iftImage *img       = iftReadImageByExt(input_image);


    /* Filter criterion: by height, area, or volume */

    char     *criterion    = iftGetJString(json, "criterion"); 
    int       threshold    = iftGetJInt(json, "threshold");
    iftImage *marker       = NULL;
    iftAdjRel *A           = NULL;

    if (strcmp(criterion,"height")==0){
      marker = iftAddValue(img,threshold);
    } else {
      if (strcmp(criterion,"area")==0){
	marker = iftFastAreaClose(img,threshold);
      } else {
	if (strcmp(criterion,"volume")==0){
	  marker = iftVolumeClose(img,threshold);
	}
      }
    }

    if (iftIs3DImage(img))
      A = iftSpheric(sqrtf(3.0));
    else
      A = iftCircular(sqrtf(2.0));
    
    aimg = iftSuperiorReconstruction(img,marker,A);

    iftWriteImageByExt(aimg->cost,output_image);

    free(input_image);
    free(output_image);
    free(criterion);
    iftDestroyImage(&marker);
    iftDestroyImage(&img);
    iftDestroyAdjRel(&A);
  }

  /* ------- Closing by reconstruction (preserves the values of the
             remaining regions). It can close the smaller basins and
             preserve the larger ones ------------ */
 
  if (strcmp(operator,"closing_reconstruction")==0){

    char     *input_image  = iftGetJString(json, "input_image");
    char     *output_image = iftGetJString(json, "output_image");
    iftImage *img          = iftReadImageByExt(input_image);
    float     adjacency_radius = (float)iftGetJDouble(json, "adjacency_radius");
    iftAdjRel *A           = NULL;

    if (iftIs3DImage(img))
      A = iftSpheric(adjacency_radius);
    else
      A = iftCircular(adjacency_radius);
    
    aimg   = iftClosingByReconstruction(img,A);

    iftWriteImageByExt(aimg->cost,output_image);

    free(input_image);
    free(output_image);
    iftDestroyImage(&img);
    iftDestroyAdjRel(&A);
  }


  /* ------- Closing of basins (preserves the values of the remaining
             regions). It closes all basins (holes) --------*/
 
  if (strcmp(operator,"closing_basins")==0){

    char     *input_image  = iftGetJString(json, "input_image");
    char     *output_image = iftGetJString(json, "output_image");
    iftImage *img          = iftReadImageByExt(input_image);

    aimg   = iftClosingOfBasins(img);

    iftWriteImageByExt(aimg->cost,output_image);

    free(input_image);
    free(output_image);
    iftDestroyImage(&img);
  }

  /* ------- Minimum Spanning Forest: flat-zone labeling ----- */
 
  if (strcmp(operator,"flat_zone_labeling")==0){

    char     *input_image  = iftGetJString(json, "input_image");
    char     *output_image = iftGetJString(json, "output_image");
    iftImage *img          = iftReadImageByExt(input_image);
    iftImage *handicap     = iftCreateImage(img->xsize,img->ysize,img->zsize);
    iftAdjRel *A;

    if (iftIs3DImage(img))
      A = iftSpheric(sqrtf(3.0));
    else
      A = iftCircular(sqrtf(2.0));

    iftSetImage(handicap,INFINITY_INT);
      
    aimg   = iftGeneralAlgorithmWithNoSeeds(img,A,handicap,iftFmsf);

    if (iftIs3DImage(img)){
      iftWriteImageByExt(aimg->label,output_image);
    } else {
      iftImage *border = iftMyDrawBorders(aimg);
      iftWriteImageByExt(border,output_image);
      iftDestroyImage(&border);
    }

    free(input_image);
    free(output_image);
    iftDestroyImage(&img);
    iftDestroyImage(&handicap);
    iftDestroyAdjRel(&A);
  }

  iftDestroyJson(&json);
  free(operator);
  iftDestroyAnnImg(&aimg);

  return(0);
}
