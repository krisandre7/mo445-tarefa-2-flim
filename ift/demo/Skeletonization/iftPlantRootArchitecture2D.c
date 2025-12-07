#include "ift.h"

void DrawBorders(iftImage *img, iftImage *bin)
{
  iftAdjRel *A   = iftCircular(sqrtf(2.0));
  iftAdjRel *B   = iftCircular(0.0);
  iftColor RGB, YCbCr;
  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 255;
  YCbCr      = iftRGBtoYCbCr(RGB,255);
  iftDrawBorders(img,bin,A,YCbCr,B);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
}

void DrawObject(iftImage *img, iftImage *bin)
{
  iftAdjRel *B   = iftCircular(0.0);
  iftColor RGB, YCbCr;
  RGB.val[0] = 255;
  RGB.val[1] = 0;
  RGB.val[2] = 255;
  YCbCr      = iftRGBtoYCbCr(RGB,255);
  iftDrawObject(img,bin,YCbCr,B);
  iftDestroyAdjRel(&B);
}

void DrawSeedPoint(iftImage *img, iftSet *S)
{
  iftAdjRel *B   = iftCircular(3.0);
  iftColor RGB, YCbCr;
  RGB.val[0] = 0;
  RGB.val[1] = 0;
  RGB.val[2] = 255;
  YCbCr      = iftRGBtoYCbCr(RGB,255);
  iftSet *aux = S;
  while (aux != NULL){
    iftVoxel u = iftGetVoxelCoord(img,aux->elem);
    iftDrawPoint(img,u,YCbCr,B);
    aux = aux->next;
  }
  iftDestroyAdjRel(&B);
}

void DrawPoints(iftImage *img, iftSet *S)
{
  iftAdjRel *B   = iftCircular(3.0);
  iftColor RGB, YCbCr;
  RGB.val[0] = 50;
  RGB.val[1] = 255;
  RGB.val[2] = 50;
  YCbCr      = iftRGBtoYCbCr(RGB,255);
  iftSet *aux = S;
  while (aux != NULL){
    iftVoxel u = iftGetVoxelCoord(img,aux->elem);
    iftDrawPoint(img,u,YCbCr,B);
    aux = aux->next;
  }
  iftDestroyAdjRel(&B);
}

iftSet *iftPlantSeed(iftImage *mat)
{
  int        pmax=IFT_NIL,Dmax=IFT_INFINITY_INT_NEG;
  iftSet     *S=NULL;

  for (int p=0; p < mat->n; p++){
    if (mat->val[p]>Dmax){
      pmax = p;
      Dmax = mat->val[p];
    }
  }
  iftInsertSet(&S,pmax);

  return(S);
}

iftImage *iftSegmentPlantRoot(iftImage *img, iftImage *marker, int thres, iftAdjRel *A)
{
  iftImage   *pathval = NULL, *pred = NULL, *label = NULL;
  iftGQueue  *Q = NULL;
  int         i, p, q, tmp, Imax = iftMaximumValue(img), V0 = 0, n = 0;
  iftVoxel    u, v;

  
  // Initialization
  
  pathval  = iftCreateImage(img->xsize, img->ysize, img->zsize);
  pred     = iftCreateImage(img->xsize, img->ysize, img->zsize);
  Q        = iftCreateGQueue(Imax + 1, img->n, pathval->val);

  for (p = 0; p < img->n; p++)
  {
    pathval->val[p] = IFT_INFINITY_INT;
    if (marker->val[p]!=0){
      V0 += img->val[p]; n++; 
      pred->val[p]    = IFT_NIL;
      pathval->val[p] = 0;
      iftInsertGQueue(&Q, p);
    }
  }
  V0 /= n; 

  // Image Foresting Transform

  while (!iftEmptyGQueue(Q))
  {
    p = iftRemoveGQueue(Q);
    u = iftGetVoxelCoord(img, p);

    for (i = 1; i < A->n; i++)
    {
      v = iftGetAdjacentVoxel(A, u, i);

      if (iftValidVoxel(img, v))
      {
        q = iftGetVoxelIndex(img, v);
	
        if (Q->L.elem[q].color != IFT_BLACK) {

	  tmp   = abs(img->val[q]-V0);

          if (tmp < pathval->val[q])  {
	    if (Q->L.elem[q].color == IFT_GRAY)
	      iftRemoveGQueueElem(Q,q);		
            pred->val[q]     = p;
            pathval->val[q]  = tmp;
            iftInsertGQueue(&Q, q);
          }
        }
      }
    }
  }

  for (p=0; p < img->n; p++) {
    if ((img->val[p]>=thres)&&(pathval->val[p]<=V0-thres+1)){
      q = p;
      while (q!=IFT_NIL){
	marker->val[q]=255;
	q = pred->val[q];
      }
    }
  }

  iftWriteImageP2(marker,"teste.pgm");
  
  label = iftCreateImage(marker->xsize,marker->ysize,marker->zsize);
  
  for (p=0; p < img->n; p++) {
    if (marker->val[p]) {
      u = iftGetVoxelCoord(img, p);
      n = 0;
      for (i = 1; i < A->n; i++){
	v = iftGetAdjacentVoxel(A, u, i);
	
	if (iftValidVoxel(img, v))
	  {
	    q = iftGetVoxelIndex(img, v);
	    if (marker->val[q]){
	      n++;
	    }
	  }
      }
      if (n > 2)
	label->val[p]=255;
    }
  }
  
  iftDestroyGQueue(&Q);
  iftDestroyImage(&pred);
  iftDestroyImage(&pathval);
  
  iftCopyVoxelSize(img, label);

  return (label);
}

int main(int argc, char *argv[]) 
{
  iftImage  *img, *aux, *mask, *skel, *mat;
  iftSet    *Tips=NULL, *Seed=NULL;
  int        thres;
  iftAdjRel *A;
   char       filename[200];
  timer     *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=4){
    iftError("Usage: iftPlantRootArchitecture2D <input.png> <output basename> <thres>","main");
  }


  img       = iftReadImageByExt(argv[1]); 
  thres     =  atoi(argv[3]); 

  t1     = iftTic();

  /* Segment the plant root image */
  
  mask    = iftThreshold(img,thres,IFT_INFINITY_INT,255);
  aux     = iftDilateBin(mask,&Seed,1.0);
  iftDestroySet(&Seed);
  iftDestroyImage(&mask);
  A      = iftCircular(sqrtf(2.0));
  mask   = iftSelectLargestComp(aux,A);
  iftDestroyImage(&aux);

  aux = iftCopyImage(img);
  DrawBorders(aux,mask);
  sprintf(filename,"%s-border.png",argv[2]);
  iftWriteImageByExt(aux,filename);    
  iftDestroyImage(&aux);  

  /* Compute its skeleton */

  mat   = iftMedialAxisTrans2D(mask, 0.1, IFT_INTERIOR);
  skel  = iftThreshold(mat,1,IFT_INFINITY_INT,255);
  aux    = iftCopyImage(img);
  DrawObject(aux,skel);
  sprintf(filename,"%s-skel.png",argv[2]);

  /* Compute terminal points and seed point */

  Tips = iftTerminalPointSet2D(skel);
  Seed = iftPlantSeed(mat);
  DrawPoints(aux,Tips);
  DrawSeedPoint(aux,Seed);
  iftWriteImageByExt(aux,filename);    
  iftDestroyImage(&aux);

  t2     = iftToc();
  fprintf(stdout,"Root representation in %f ms\n",iftCompTime(t1,t2));

  iftDestroyAdjRel(&A);
  iftDestroyImage(&img);
  iftDestroyImage(&mask);
  iftDestroyImage(&skel);
  iftDestroyImage(&mat);
  iftDestroySet(&Tips);
  iftDestroySet(&Seed);
  
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);

}
