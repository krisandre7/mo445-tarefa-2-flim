#include "ift.h"

void  SaveResults(iftImage *img, iftImage *label, char *basename)
{
  char     filename[200];

  if (iftIs3DImage(img)){
    sprintf(filename,"%s-label.scn",basename);
    iftWriteImageByExt(label,filename);
  } else {
    sprintf(filename,"%s-label.png",basename);
    iftWriteImageByExt(label,filename);

    iftColor   RGB, YCbCr;
    int        normvalue = iftNormalizationValue(iftMaximumValue(img));
    iftAdjRel *A = iftCircular(1.0), *B = iftCircular(0.0);
    RGB.val[0]   = normvalue;
    RGB.val[1]   = normvalue/3.0;
    RGB.val[2]   = 0;    
    YCbCr        = iftRGBtoYCbCr(RGB, normvalue);
    sprintf(filename,"%s-borders.png", basename);
    iftDrawBorders(img,label,A,YCbCr,B);
    iftWriteImageByExt(img, filename);
    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
  }
}

float iftArcWeightByPathMean(iftMImage *mimg,iftImage *pred,int p,int q,iftGQueue *Q, int npredecessors)
{
  float    mean[mimg->m], nelems=1, weight=0.0;
  
  for (int m=0; m < mimg->m; m++)
    mean[m]=mimg->val[p][m];

  p = pred->val[p];
  while ((p != IFT_NIL)&&(nelems <= npredecessors)) {
    for (int m=0; m < mimg->m; m++){
      mean[m] += mimg->val[p][m];
      nelems++;
    }
    p = pred->val[p];
  }

  for (int m=0; m < mimg->m; m++)
    mean[m]=mean[m]/nelems;

  for (int m=0; m < mimg->m; m++){
    weight += iftPowerOfTwo(mimg->val[q][m]-mean[m]);
  }

  return(sqrtf(weight));
}

iftImage *iftSeedCompByPathMean(iftImage *img, iftLabeledSet *S, int npredecessors)
{
  iftImage      *cost = NULL, *label = NULL, *pred = NULL;
  iftMImage     *mimg = NULL;
  iftGQueue     *Q    = NULL;
  int            i, p, q, tmp, Imax = iftMaximumValue(img);
  float          weight;
  iftVoxel       u, v;
  iftLabeledSet *seeds=S;
  iftAdjRel     *A = NULL;

  // Initialization

  if (iftIs3DImage(img)) {
    A = iftSpheric(1.0);
  } else {
    A = iftCircular(1.0);
  }

  if (iftIsColorImage(img)){
    mimg   = iftImageToMImage(img,LABNorm2_CSPACE);
  } else {
    mimg   = iftImageToMImage(img,GRAYNorm_CSPACE);
  }

  cost   = iftCreateImage(img->xsize, img->ysize, img->zsize);
  label  = iftCreateImage(img->xsize, img->ysize, img->zsize);
  pred   = iftCreateImage(img->xsize, img->ysize, img->zsize);
  Q      = iftCreateGQueue(Imax, cost->n, cost->val);
  iftSetImage(cost,IFT_INFINITY_INT);

  while (seeds != NULL)
  {
    p               = seeds->elem;
    label->val[p]   = seeds->label;
    pred->val[p]    = IFT_NIL;
    cost->val[p]    = 0;
    iftInsertGQueue(&Q, p);
    seeds = seeds->next;
  }


  // Image Foresting Transform

  while (!iftEmptyGQueue(Q))
  {
    p = iftRemoveGQueue(Q);
    u = iftGetVoxelCoord(cost, p);

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);

      if (iftValidVoxel(cost, v))
      {
        q = iftGetVoxelIndex(cost, v);

	if (Q->L.elem[q].color != IFT_BLACK){
	  {
	    weight      = iftArcWeightByPathMean(mimg,pred,p,q,Q,npredecessors);	    
	    tmp         = iftMax(cost->val[p],iftRound(Imax*weight));
	    if (tmp < cost->val[q])  
	      {
		if (Q->L.elem[q].color == IFT_GRAY){
		  iftRemoveGQueueElem(Q,q);
		}
		label->val[q]    = label->val[p];
		pred->val[q]     = p;
		cost->val[q]     = tmp;
		iftInsertGQueue(&Q, q);
	      }
	  }
	}
      }
    }
  }
  
  iftDestroyAdjRel(&A);
  iftDestroyGQueue(&Q);
  iftDestroyImage(&cost);
  iftDestroyImage(&pred);
  iftDestroyMImage(&mimg);
  
  iftCopyVoxelSize(img, label);

  return(label);
}

int main(int argc, char *argv[]) 
{
  iftImage      *img, *label;
  iftLabeledSet *S=NULL;
  int            npredecessors;
  int            MemDinInicial, MemDinFinal;
  timer         *tstart=NULL;

  MemDinInicial = iftMemoryUsed(1);

  if (argc != 5)
    iftError("Usage: iftSeedCompByPathMean <input-image.*> <input-seeds.txt> <npredecessors> <output-basename>","main");

  img     = iftReadImageByExt(argv[1]);
  S       = iftReadSeeds(argv[2],img);
  npredecessors = atoi(argv[3]);
  
  tstart = iftTic();

  label = iftSeedCompByPathMean(img, S, npredecessors);

  puts("\nDone...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));

  SaveResults(img,label,argv[4]);
  
  iftDestroyImage(&img);
  iftDestroyImage(&label);
  iftDestroyLabeledSet(&S);
  
  MemDinFinal = iftMemoryUsed();
  iftVerifyMemory(MemDinInicial, MemDinFinal);

  return(0);
}

/* Unused functions: the tests did not show any advantage in using them. */

int iftRelaxationLabel(iftImage *basins,iftImage *label,int nlabels, int p, iftAdjRel *B, iftGQueue *Q)
{
  float weight[B->n], sum[nlabels+1];
  iftVoxel u = iftGetVoxelCoord(basins,p);
  float max_membership = IFT_INFINITY_FLT_NEG;
  int   max_label = IFT_NIL;
  
  for (int i=0; i < B->n; i++) {
    iftVoxel v = iftGetAdjacentVoxel(B,u,i);
    weight[i]=0;
    if (iftValidVoxel(label,v)){
      int q = iftGetVoxelIndex(label,v);
      if (Q->L.elem[q].color!=IFT_WHITE){  
        weight[i] = 1.0 / (1.0 + (0.5 * basins->val[q])); // pseudo inverse
      }
    }
  }
  
  for (int l = 0; l <= nlabels; l++)
    sum[l] = 0.0;

  for (int i=0; i < B->n; i++) {
    iftVoxel v = iftGetAdjacentVoxel(B,u,i);
    if (iftValidVoxel(label,v)){
      int q = iftGetVoxelIndex(label,v);
      if (Q->L.elem[q].color!=IFT_WHITE){  
	sum[label->val[q]] += weight[i];
      }
    }
  }

  max_membership = IFT_INFINITY_FLT_NEG;
  max_label      = IFT_NIL;
  for (int l = 0; l <= nlabels; l++){
    if (sum[l] > max_membership) {
      max_membership = sum[l];
      max_label      = l;
    }
  }
    
  return(max_label);
}

float iftArcWeightByLocalMean(iftMImage *mimg,iftImage *label,int p,int q,iftAdjRel *B,iftGQueue *Q)
{
  float    mean[mimg->m], nelems=1, weight=0.0;
  iftVoxel u = iftGetVoxelCoord(label,p);
   
  for (int m=0; m < mimg->m; m++)
    mean[m]=mimg->val[p][m];

  for (int i=1; i < B->n; i++) {
    iftVoxel vt = iftGetAdjacentVoxel(B,u,i); 
    if (iftValidVoxel(label,vt)){
      int t = iftGetVoxelIndex(label,vt);
      if ((label->val[t]==label->val[p])&&(Q->L.elem[t].color!=IFT_WHITE)){
	for (int m=0; m < mimg->m; m++){
	  mean[m] += mimg->band[m].val[t];
	  nelems++;
	}
      }
    }
  }

  for (int m=0; m < mimg->m; m++)
    mean[m]=mean[m]/nelems;

  for (int m=0; m < mimg->m; m++){
    weight += iftPowerOfTwo(mimg->val[q][m]-mean[m]);
  }

  return(sqrtf(weight));
}


iftImage *iftSeedCompByLocalMean(iftImage *img, iftLabeledSet *S, float radius)
{
  iftImage      *cost = NULL, *label = NULL;
  iftMImage     *mimg = NULL;
  iftGQueue     *Q = NULL;
  int            i, p, q, tmp, Imax = iftMaximumValue(img);
  float          weight;
  iftVoxel       u, v;
  iftLabeledSet *seeds=S;
  iftAdjRel     *A = NULL, *B = NULL;

  // Initialization

  if (iftIs3DImage(img)) {
    A = iftSpheric(1.0);
    B = iftSpheric(radius);
  } else {
    A = iftCircular(1.0);
    B = iftCircular(radius);
  }

  if (iftIsColorImage(img)){
    mimg   = iftImageToMImage(img,LABNorm2_CSPACE);
  } else {
    mimg   = iftImageToMImage(img,GRAYNorm_CSPACE);
  }

  cost  = iftCreateImage(img->xsize, img->ysize, img->zsize);
  label = iftCreateImage(img->xsize, img->ysize, img->zsize);
  Q     = iftCreateGQueue(Imax, cost->n, cost->val);
  iftSetImage(cost,IFT_INFINITY_INT);

  while (seeds != NULL)
  {
    p               = seeds->elem;
    label->val[p]   = seeds->label;
    cost->val[p]    = 0;
    iftInsertGQueue(&Q, p);
    seeds = seeds->next;
  }


  // Image Foresting Transform

  while (!iftEmptyGQueue(Q))
  {
    p = iftRemoveGQueue(Q);
    u = iftGetVoxelCoord(cost, p);

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);

      if (iftValidVoxel(cost, v))
      {
        q = iftGetVoxelIndex(cost, v);

	if (Q->L.elem[q].color != IFT_BLACK){
	  {
	    weight = iftArcWeightByLocalMean(mimg,label,p,q,B,Q);	    
	    tmp    = iftMax(cost->val[p], iftRound(Imax*weight));

	    if (tmp < cost->val[q])  
	      {
		if (Q->L.elem[q].color == IFT_GRAY){
		  iftRemoveGQueueElem(Q,q);
		}
		label->val[q]    = label->val[p];
		cost->val[q]     = tmp;
		iftInsertGQueue(&Q, q);
	      }
	  }
	}
      }
    }
  }
  
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  iftDestroyGQueue(&Q);
  iftDestroyImage(&cost);
  iftDestroyMImage(&mimg);
  
  iftCopyVoxelSize(img, label);

  return(label);
}

