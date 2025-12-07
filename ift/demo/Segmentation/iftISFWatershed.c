#include "ift.h"

iftLabeledSet *iftRecomputeSeeds(iftImage *basins, iftImage *regmin, iftImage *label)
{
  iftLabeledSet *S1 = iftGeometricCenters(label);
  iftLabeledSet *S2 = NULL;
  iftAdjRel     *A=NULL;
  int            nseeds = iftMaximumValue(label);

  if (iftIs3DImage(basins)){
    float superpixelsize = 0.5+(float)(basins->n)/(float)nseeds;
    float step = (float) pow((double)superpixelsize,1.0/3.0)+0.5;
    float radius = step/2.0;
    A = iftSpheric(radius);
  }else{
    float superpixelsize = 0.5+(float)(basins->n)/(float)nseeds;
    float step = (float) sqrt((double)superpixelsize)+0.5;
    float radius = step/2.0;
    A = iftCircular(radius);
  }
  
  int newlabel=1;
  while (S1 != NULL) {
    int lb;
    int p = iftRemoveLabeledSet(&S1,&lb);
    iftVoxel u = iftGetVoxelCoord(basins,p); 
    for (int i = 0; i < A->n; i++) {
      iftVoxel v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(basins,v)){
	int q      = iftGetVoxelIndex(basins,v);
	if (regmin->val[q]){
	  iftInsertLabeledSet(&S2,q,newlabel);
	  newlabel++;
	  break;
	}
      }
    }
  }

  iftDestroyAdjRel(&A);
  return(S2);
}

iftImage *iftISFWatershed(iftImage *basins, iftAdjRel *A, int handicap, iftLabeledSet *seeds) {
  iftImage  *pathval = NULL, *label = NULL, *root = NULL;
  iftGQueue  *Q = NULL;
  int      i, p, q, tmp;
  iftVoxel    u, v;
  iftLabeledSet *S = seeds;

  // Initialization
  pathval  = iftCreateImage(basins->xsize, basins->ysize, basins->zsize);
  label    = iftCreateImage(basins->xsize, basins->ysize, basins->zsize);
  root     = iftCreateImage(basins->xsize, basins->ysize, basins->zsize);
  Q        = iftCreateGQueue(iftMaximumValue(basins)+handicap+1, basins->n, pathval->val);

  for (p = 0; p < basins->n; p++)
  {
    pathval->val[p] = IFT_INFINITY_INT;
  }

  while (S != NULL)
  {
    p = S->elem;
    u = iftGetVoxelCoord(basins,p);
    root->val[p]  = p;
    label->val[p] = S->label;
    pathval->val[p] = basins->val[p]+handicap;
    iftInsertGQueue(&Q, p);
    S = S->next;
  }

  // Image Foresting Transform

  while (!iftEmptyGQueue(Q))
  {
    p = iftRemoveGQueue(Q);
    u = iftGetVoxelCoord(basins, p);

    if (root->val[p]==p)
      pathval->val[p] = basins->val[p];

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);

      if (iftValidVoxel(basins, v))
      {
        q = iftGetVoxelIndex(basins, v);
        if (pathval->val[q] > pathval->val[p])
        {	  
          tmp = iftMax(pathval->val[p], basins->val[q]);

          if (tmp < pathval->val[q])  // For this path-value function,
          {
	    if (Q->L.elem[q].color == IFT_GRAY)
	      iftRemoveGQueueElem(Q,q);
            label->val[q]    = label->val[p];
            root->val[q]     = root->val[p];
            pathval->val[q]  = tmp;
            iftInsertGQueue(&Q, q);
          }
        }
      }
    }
  }

  iftDestroyGQueue(&Q);

  iftDestroyImage(&pathval);
  iftDestroyImage(&root);

  iftCopyVoxelSize(basins, label);

  return (label);
}


iftLabeledSet *iftSelectWaterSeeds(iftImage *basins, iftImage *regmin, iftImage *seeds)
{
  iftLabeledSet *S1 = NULL, *S2=NULL;
  iftAdjRel     *A=NULL;
  int            nseeds = iftNumberOfElements(seeds);

  if (iftIs3DImage(basins)){
    float superpixelsize = 0.5+(float)(basins->n)/(float)nseeds;
    float step = (float) pow((double)superpixelsize,1.0/3.0)+0.5;
    float radius = step/2.0;
    A = iftSpheric(radius);
  }else{
    float superpixelsize = 0.5+(float)(basins->n)/(float)nseeds;
    float step = (float) sqrt((double)superpixelsize)+0.5;
    float radius = step/2.0;
    A = iftCircular(radius);
  }
    
  for (int p=0, l=1; p < seeds->n; p++) 
    if (seeds->val[p]){
      iftInsertLabeledSet(&S1,p,l);
      l++;
    }

  while (S1 != NULL) {
    int label;
    int p = iftRemoveLabeledSet(&S1,&label);
    iftVoxel u = iftGetVoxelCoord(basins,p); 
    for (int i = 0; i < A->n; i++) {
      iftVoxel v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(basins,v)){
	int q      = iftGetVoxelIndex(basins,v);
	if (regmin->val[q]){
	  iftInsertLabeledSet(&S2,q,label);
	  break;
	}
      }
    }
  }

  iftDestroyAdjRel(&A);

  return(S2);
}


int main(int argc, char *argv[]) 
{
  iftImage        *img=NULL, *basins=NULL, *label=NULL;
  iftImage        *regmin=NULL;
  iftMImage       *mimg;
  iftAdjRel       *A=NULL;
  timer           *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=5)
    iftError("Usage: iftISFWatershed <image.[pgm,ppm,scn]> <grad_radius> <nsuperpixels> <handicap>","main");
  
  img   = iftReadImageByExt(argv[1]);    
  
  t1 = iftTic();

  if (iftIsColorImage(img)){
    mimg   = iftImageToMImage(img,LABNorm_CSPACE);
  } else {
    mimg   = iftImageToMImage(img,GRAY_CSPACE);
  }

  if (iftIs3DImage(img))
    A      = iftSpheric(atof(argv[2]));
  else 
    A      = iftCircular(atof(argv[2]));

  basins           = iftMImageBasins(mimg, A);
  regmin           = iftRegionalMinima(basins);

  iftDestroyAdjRel(&A);

  if (iftIs3DImage(img))
    A      = iftSpheric(sqrtf(3.0));
  else 
    A      = iftCircular(sqrtf(2.0));
  
  int nsuperpixels = atoi(argv[3]);
  iftImage *mask = iftSelectImageDomain(mimg->xsize,mimg->ysize,mimg->zsize);
  //iftImage *seeds   = iftGridSampling(mimg,mask,nsuperpixels);
  iftImage *seeds   = iftAltMixedSampling(mimg,mask,nsuperpixels);
  iftDestroyImage(&mask);
  iftDestroyMImage(&mimg);


  iftLabeledSet *S = iftSelectWaterSeeds(basins, regmin, seeds);
  iftDestroyImage(&seeds);
  label  = iftISFWatershed(basins, A, atoi(argv[4]), S);
  iftDestroyLabeledSet(&S);

  /* for (int iter=1; iter <= 3; iter++) { */
  /*   S = iftRecomputeSeeds(basins, regmin, label); */
  /*   iftDestroyImage(&label); */
  /*   label  = iftISFWatershed(basins, A, atoi(argv[4]), S); */
  /*   iftDestroyLabeledSet(&S); */
  /* } */

  printf("%d\n",iftMaximumValue(label));
  iftImage *nlabel = iftSmoothRegionsByDiffusion(label, img, 0.5, 2);
  iftDestroyImage(&label);
  label = iftRelabelRegions(nlabel,A);
  iftDestroyImage(&nlabel);

  iftDestroyImage(&basins);  
  iftDestroyImage(&regmin);  
 
  t2     = iftToc(); 

  fprintf(stdout,"ISFWatershed in %f ms with %d regions\n",iftCompTime(t1,t2),iftMaximumValue(label));

  if (!iftIs3DImage(img)){
    iftWriteImageP2(label,"labels.pgm"); 
    iftAdjRel   *B;
    iftColor     RGB,YCbCr;
    RGB.val[0] = 255;
    RGB.val[1] = 0;
    RGB.val[2] = 0;
    YCbCr      = iftRGBtoYCbCr(RGB,255);
    iftDestroyAdjRel(&A); 
    A          = iftCircular(sqrtf(2.0));
    B          = iftCircular(0.0);
    iftDrawBorders(img,label,A,YCbCr,B);
    iftWriteImageP6(img,"result.ppm");
    iftDestroyAdjRel(&B);
  }

  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);  
  iftDestroyImage(&label);  
  
  /* ---------------------------------------------------------- */
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}

