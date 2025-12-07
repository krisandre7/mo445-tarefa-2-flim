#include "ift.h"

/* Implemented for the book chapter only */

iftImage       *iftAdvectionField(iftImage *bin, iftAdjRel *A, iftImage **edt);

iftImage *iftAdvectionField(iftImage *bin, iftAdjRel *A, iftImage **edt)
{
  iftImage *dist=NULL,*root=NULL,*pred=NULL;
  iftGQueue *Q=NULL;
  int i,p,q,tmp;
  iftVoxel u,v,r;
  iftSet *S=NULL;
  iftAdjRel *B;

  // Initialization

  if (iftIs3DImage(bin)) {
    B = iftSpheric(1.0);
  }
  else { 
    B = iftCircular(1.0);
  }

  dist  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
  root  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
  pred  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
  Q     = iftCreateGQueue(QSIZE,bin->n,dist->val);

  for (p=0; p < bin->n; p++) {
    if (bin->val[p] > 0)
      dist->val[p]= IFT_INFINITY_INT;
  }

  S     = iftObjectBorderSet(bin, B); // seeds
  iftDestroyAdjRel(&B);
  while (S != NULL) {
    p = iftRemoveSet(&S);
    dist->val[p]=0;
    root->val[p]=p;
    pred->val[p]=IFT_NIL;
    iftInsertGQueue(&Q,p);
  }
  
  // Image Foresting Transform

  while(!iftEmptyGQueue(Q)) {

    p=iftRemoveGQueue(Q);
    //Gets the voxel and its root.
    u = iftGetVoxelCoord(bin,p);
    r = iftGetVoxelCoord(bin,root->val[p]);

    for (i=1; i < A->n; i++){
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(bin,v)){
	q = iftGetVoxelIndex(bin,v);
	if (dist->val[q] > dist->val[p]){
	  tmp = iftSquaredVoxelDistance(v,r);
	  if (tmp < dist->val[q]){
	    if (dist->val[q] != IFT_INFINITY_INT)
	      iftRemoveGQueueElem(Q, q);
	    dist->val[q]  = tmp;
	    root->val[q]  = root->val[p];
	    pred->val[q]  = p;
	    iftInsertGQueue(&Q, q);
	  }
	}
      }
    }
  }

  iftDestroyGQueue(&Q);
  iftDestroyImage(&root);
  *edt = dist;
  iftCopyVoxelSize(bin,pred);
  return(pred);
}

int main(int argc, char *argv[]) 
{
  iftImage       *bin=NULL,*pred=NULL,*edt=NULL,*aux=NULL,*skel=NULL;
  iftFImage      *faux=NULL, *msskel=NULL;
  iftAdjRel      *A=NULL,*B=NULL,*C=NULL;
  int             p[2];
  iftVoxel        u[2], root;
  iftColor        RGB, YCbCr;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=4)
    iftError("Usage: iftAdvection <binary.pgm> <x0> <y0> (e.g. x0=93,y0=59 is a skeleton point of polygon.pgm)","main");

  A      = iftCircular(sqrtf(2.0)); 
  B      = iftCircular(1.0); 
  C      = iftCircular(0.0); 

  aux    = iftReadImageP5(argv[1]);
  bin    = iftThreshold(aux,127,255,255);
  iftDestroyImage(&aux);

  pred   = iftAdvectionField(bin,A,&edt);
  msskel = iftMSSkel2D(bin,A,IFT_INTERIOR);
  iftDestroyImage(&bin);
  iftDestroyImage(&aux);

  float maxval = iftFMaximumValue(msskel);
  for(p[0]=0; p[0] < msskel->n; p[0]++) 
    if (fabs(msskel->val[p[0]]-maxval)<IFT_EPSILON){
      root = iftFGetVoxelCoord(msskel,p[0]);
      break;
    }
  skel   = iftFThreshold(msskel,5.0,100.0,1);
  iftDestroyFImage(&msskel);

  faux   = iftSQRT(edt);
  iftDestroyImage(&edt);
  edt    = iftFImageToImage(faux,255);
  iftDestroyFImage(&faux);

  RGB.val[0]=255;
  RGB.val[1]=100;
  RGB.val[2]=0;
  YCbCr     = iftRGBtoYCbCr(RGB,255);
  aux       = iftRootVoxels(pred); 
  iftDrawObject(edt,aux,YCbCr,B);
  iftDestroyImage(&aux);
  
  RGB.val[0]=255;
  RGB.val[1]=255;
  RGB.val[2]=0;
  YCbCr     = iftRGBtoYCbCr(RGB,255);
  aux       = iftLeafVoxels(pred,A);
  iftDrawObject(edt,aux,YCbCr,C);
  iftDestroyImage(&aux);

  RGB.val[0]=50;
  RGB.val[1]=255;
  RGB.val[2]=255;
  YCbCr     = iftRGBtoYCbCr(RGB,255);
  iftDrawObject(edt,skel,YCbCr,C);

  RGB.val[0]=255;
  RGB.val[1]=0;
  RGB.val[2]=255;
  YCbCr     = iftRGBtoYCbCr(RGB,255);
  u[0].x = atoi(argv[2]);
  u[0].y = atoi(argv[3]);
  u[0].z = 0;
  p[0]   = iftGetVoxelIndex(edt,u[0]);
  p[1]   = p[0];
  while (p[1] != IFT_NIL) {
    u[1] = iftGetVoxelCoord(edt,p[1]);
    iftDrawPoint(edt,u[1],YCbCr,C);
    p[1] = pred->val[p[1]];
  }

  RGB.val[0]=255;
  RGB.val[1]=0;
  RGB.val[2]=255;
  YCbCr     = iftRGBtoYCbCr(RGB,255);
  u[0].x = atoi(argv[2])-1;
  u[0].y = atoi(argv[3]);
  u[0].z = 0;
  p[0]   = iftGetVoxelIndex(edt,u[0]);
  p[1]   = p[0];
  while (p[1] != IFT_NIL) {
    u[1] = iftGetVoxelCoord(edt,p[1]);
    iftDrawPoint(edt,u[1],YCbCr,C);
    p[1] = pred->val[p[1]];
  }

  RGB.val[0]=255;
  RGB.val[1]=0;
  RGB.val[2]=255;
  YCbCr     = iftRGBtoYCbCr(RGB,255);
  u[0].x = atoi(argv[2])+1;
  u[0].y = atoi(argv[3]);
  u[0].z = 0;
  p[0]   = iftGetVoxelIndex(edt,u[0]);
  p[1]   = p[0];
  while (p[1] != IFT_NIL) {
    u[1] = iftGetVoxelCoord(edt,p[1]);
    iftDrawPoint(edt,u[1],YCbCr,C);
    p[1] = pred->val[p[1]];
  }

  RGB.val[0]=255;
  RGB.val[1]=0;
  RGB.val[2]=255;
  YCbCr     = iftRGBtoYCbCr(RGB,255);
  u[0].x = atoi(argv[2]);
  u[0].y = atoi(argv[3])-1;
  u[0].z = 0;
  p[0]   = iftGetVoxelIndex(edt,u[0]);
  p[1]   = p[0];
  while (p[1] != IFT_NIL) {
    u[1] = iftGetVoxelCoord(edt,p[1]);
    iftDrawPoint(edt,u[1],YCbCr,C);
    p[1] = pred->val[p[1]];
  }

  RGB.val[0]=255;
  RGB.val[1]=0;
  RGB.val[2]=255;
  YCbCr     = iftRGBtoYCbCr(RGB,255);
  u[0].x = atoi(argv[2]);
  u[0].y = atoi(argv[3])+1;
  u[0].z = 0;
  p[0]   = iftGetVoxelIndex(edt,u[0]);
  p[1]   = p[0];
  while (p[1] != IFT_NIL) {
    u[1] = iftGetVoxelCoord(edt,p[1]);
    iftDrawPoint(edt,u[1],YCbCr,C);
    p[1] = pred->val[p[1]];
  }

  RGB.val[0]=255;
  RGB.val[1]=0;
  RGB.val[2]=0;
  YCbCr     = iftRGBtoYCbCr(RGB,255);
  iftDrawPoint(edt,root,YCbCr,A);
  
  iftWriteImageP6(edt,"advection-path.ppm");
  iftDestroyImage(&edt);
  iftDestroyImage(&pred);
  iftDestroyImage(&skel);
  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
 
  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);
}



