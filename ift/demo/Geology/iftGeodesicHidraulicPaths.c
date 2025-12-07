#include "ift.h"

iftSet *iftSelectSeed(iftImage *bin)
{
  iftSet *S=NULL;
  iftVoxel u;
  int p, ym=0, n=0;

  u.z = u.x = 0;
  for (u.y=0; u.y < bin->ysize; u.y++) {
    p = iftGetVoxelIndex(bin,u);
    if (bin->val[p]!=0){
      n++;
      ym += u.y;
    }      
  }
  u.y = ym/n;
  p   = iftGetVoxelIndex(bin,u);
  iftInsertSet(&S,p);
  return(S);
}


iftImage  *iftGeodesicPath(iftImage *bin, iftAdjRel *A, iftSet *S)
{
  iftImage  *dist=NULL,*pred=NULL;
  iftGQueue *Q=NULL;
  int        i,p,q,tmp, closest=IFT_NIL;
  iftVoxel   u,v;
  iftSet    *seed;

  dist  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
  pred  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
  Q     = iftCreateGQueue(QSIZE,bin->n,dist->val);

  for (p=0; p < bin->n; p++) {
    if (bin->val[p]!=0)
      dist->val[p]= IFT_INFINITY_INT;
    else
      dist->val[p]= 0;
  }

  seed = S;
  while (seed != NULL) {
    p = seed->elem;
    dist->val[p]=0;
    pred->val[p]=IFT_NIL;
    if (bin->val[p]!=0)
      iftInsertGQueue(&Q,p);
    else
      iftError("Seed is expected inside the object","iftGeodesicPath");
    seed = seed->next;
  }
  
  // Image Foresting Transform

  while(!iftEmptyGQueue(Q)) {

    p=iftRemoveGQueue(Q);

    u = iftGetVoxelCoord(bin,p);

    if ((closest == IFT_NIL)&&(u.x == bin->xsize-1)) /* salve o ponto mais próximo do lado oposto */
      closest = p;

    for (i=1; i < A->n; i++){
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(bin,v)){
	q = iftGetVoxelIndex(bin,v);
	if (dist->val[q] > dist->val[p]){

	  if (iftSquaredVoxelDistance(u,v)==1)
	    tmp = dist->val[p] + 5;
	  else // distancia é 2 então
	    tmp = dist->val[p] + 7;

	  if (tmp < dist->val[q]){
	    if (dist->val[q] != IFT_INFINITY_INT)
	      iftRemoveGQueueElem(Q, q);
	    dist->val[q]  = tmp;
	    pred->val[q]  = p;
	    iftInsertGQueue(&Q, q);
	  }
	}
      }
    }
  }

  iftWriteImageByExt(dist,"GeodesicDist.pgm");

  if (closest==IFT_NIL)
      iftError("Terminal point could not be found inside the object","iftGeodesicPath");

  iftFColor normRGB;
  normRGB.val[0] = 0.0;
  normRGB.val[1] = 0.0;
  normRGB.val[2] = 1.0; 
  iftImage *paths = iftDrawPath(dist,pred,closest,normRGB,A);

  iftDestroyImage(&dist);
  iftDestroyGQueue(&Q);
  iftDestroyImage(&pred);

  iftCopyVoxelSize(bin,paths);
  return(paths);
}


iftImage  *iftHidraulicPath(iftImage *bin, iftImage *edt, iftAdjRel *A, iftSet *S)
{
  iftImage  *dist=NULL,*pred=NULL;
  iftGQueue *Q=NULL;
  int        i,p,q,tmp, closest=IFT_NIL, Dmax = iftMaximumValue(edt);
  iftVoxel   u,v;
  iftSet    *seed;

  dist  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
  pred  = iftCreateImage(bin->xsize,bin->ysize,bin->zsize);
  Q     = iftCreateGQueue(QSIZE,bin->n,dist->val);

  for (p=0; p < bin->n; p++) {
    if (bin->val[p]!=0)
      dist->val[p]= IFT_INFINITY_INT;
    else
      dist->val[p]= 0;
  }

  seed = S;
  while (seed != NULL) {
    p = seed->elem;
    dist->val[p]=0;
    pred->val[p]=IFT_NIL;
    if (bin->val[p]!=0)
      iftInsertGQueue(&Q,p);
    else
      iftError("Seed is expected inside the object","iftGeodesicPath");
    seed = seed->next;
  }
  
  // Image Foresting Transform

  while(!iftEmptyGQueue(Q)) {

    p=iftRemoveGQueue(Q);

    u = iftGetVoxelCoord(bin,p);

    if ((closest == IFT_NIL)&&(u.x == bin->xsize-1)) /* salve o ponto mais próximo do lado oposto */
      closest = p;

    for (i=1; i < A->n; i++){
      v = iftGetAdjacentVoxel(A,u,i);
      if (iftValidVoxel(bin,v)){
	q = iftGetVoxelIndex(bin,v);
	if (dist->val[q] > dist->val[p]){

	  if (iftSquaredVoxelDistance(u,v)==1)
	    tmp = dist->val[p] + 5*((float)edt->val[q]/Dmax);
	  else // distancia é 2 então
	    tmp = dist->val[p] + 7*((float)edt->val[q]/Dmax);

	  if (tmp < dist->val[q]){
	    if (dist->val[q] != IFT_INFINITY_INT)
	      iftRemoveGQueueElem(Q, q);
	    dist->val[q]  = tmp;
	    pred->val[q]  = p;
	    iftInsertGQueue(&Q, q);
	  }
	}
      }
    }
  }


  iftWriteImageByExt(dist,"HidraulicDist.pgm");

  if (closest==IFT_NIL)
      iftError("Terminal point could not be found inside the object","iftGeodesicPath");

  iftFColor normRGB;
  normRGB.val[0] = 1.0;
  normRGB.val[1] = 0.0;
  normRGB.val[2] = 0.0; 
  iftImage *paths = iftDrawPath(dist,pred,closest,normRGB,A);

  iftDestroyImage(&dist);
  iftDestroyGQueue(&Q);
  iftDestroyImage(&pred);

  iftCopyVoxelSize(bin,paths);
  return(paths);
}


int main(int argc, char *argv[]) 
{
  iftImage  *img[3];
  iftAdjRel *A;
  iftSet    *S=NULL;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=2){
    iftError("Usage: iftGeodesicHidraulicPaths <input.png>","main");
  }

  img[0] = iftReadImageByExt(argv[1]);
  A      = iftCircular(sqrtf(2.0));

  img[1] = iftEuclDistTrans(img[0],A,IFT_INTERIOR, NULL, NULL, NULL);

  iftWriteImageByExt(img[1],"EDT.pgm");

  img[2] = iftComplement(img[1]);

  iftWriteImageByExt(img[2],"CompEDT.pgm");

  iftDestroyImage(&img[1]);
  S      = iftSelectSeed(img[0]);
  img[1] = iftGeodesicPath(img[0],A,S);
  
  iftWriteImageByExt(img[1],"GeodesicPath.ppm");

  iftDestroyImage(&img[1]);

  img[1] = iftHidraulicPath(img[0],img[2],A,S);

  iftWriteImageByExt(img[1],"HidraulicPath.ppm");
  
  iftDestroySet(&S);
  iftDestroyAdjRel(&A);
  iftDestroyImage(&img[0]);
  iftDestroyImage(&img[1]);
  iftDestroyImage(&img[2]);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);

}
