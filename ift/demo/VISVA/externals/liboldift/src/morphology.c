#include "morphology.h"
#include "geometry.h"
#include "queue.h"
#include "mathematics.h"
#include "heap.h"

Image *Dilate(Image *img, AdjRel *A)
{
  Image *fimg=NULL,*fdil=NULL,*dil=NULL;
  AdjPxl *N=NULL;
  int sz,p,q,max,i,x,y,po;

  sz  = FrameSize(A);
  fimg = AddFrame(img,sz,INT_MIN);
  fdil = CreateImage(fimg->ncols,fimg->nrows);
  N  = AdjPixels(fimg,A);
  for (y=0,po=sz+fimg->tbrow[sz]; y < img->nrows; y++, po=po+fimg->ncols) 
    for (x=0,p=po; x < img->ncols; x++,p++){
      max = INT_MIN;
      for (i=0; i < N->n; i++){
	q = p + N->dp[i];
	if (fimg->val[q] > max)
	  max = fimg->val[q];
      }
      fdil->val[p]=max;
    }

  dil = RemFrame(fdil,sz);
  DestroyImage(&fimg);
  DestroyImage(&fdil);
  DestroyAdjPxl(&N);

  return(dil);
}

Image *Erode(Image *img, AdjRel *A)
{
  Image *fimg=NULL,*fero=NULL,*ero=NULL;
  AdjPxl *N=NULL;
  int sz,p,q,min,i,x,y,po;

  sz  = FrameSize(A);
  fimg = AddFrame(img,sz,INT_MAX);
  fero = CreateImage(fimg->ncols,fimg->nrows);
  N  = AdjPixels(fimg,A);
  for (y=0,po=sz+fimg->tbrow[sz]; y < img->nrows; y++, po=po+fimg->ncols) 
    for (x=0,p=po; x < img->ncols; x++,p++){
      min = INT_MAX;
      for (i=0; i < N->n; i++){
	q = p + N->dp[i];
	if (fimg->val[q] < min)
	  min = fimg->val[q];
      }
      fero->val[p]=min;
    }

  ero = RemFrame(fero,sz);
  DestroyImage(&fimg);
  DestroyImage(&fero);
  DestroyAdjPxl(&N);

  return(ero);
}

Image *Open(Image *img, AdjRel *A)
{
  Image *open=NULL,*ero=NULL;

  ero  = Erode(img,A);
  open = Dilate(ero,A);
  DestroyImage(&ero);

  return(open);
}

Image *Close(Image *img, AdjRel *A)
{
  Image *close=NULL,*dil=NULL;

  dil   = Dilate(img,A);
  close = Erode(dil,A);
  DestroyImage(&dil);

  return(close);
}

Image *MorphGrad(Image *img, AdjRel *A)
{
  Image *dil=NULL,*ero=NULL,*grad=NULL;

  dil  = Dilate(img,A);
  ero  = Erode(img,A);
  grad = Diff(dil,ero);
  DestroyImage(&dil);
  DestroyImage(&ero);

  return(grad);
}

Image *AsfOC(Image *img, AdjRel *A)
{
  Image *open=NULL,*close=NULL;

  open  = Open(img,A);
  close = Close(open,A);
  DestroyImage(&open);

  return(close);
}

Image *AsfnOC(Image *img, int ntimes)
{
  Image *asfoc=NULL,*aux=NULL; 
  AdjRel *A;
  int i;
  float r=1.0;
  
  aux = ift_CopyImage(img);
  for (i=1; i <= ntimes; i++){    
    A = Circular(r);
    asfoc = AsfOC(aux,A);
    DestroyAdjRel(&A);
    DestroyImage(&aux);
    aux = asfoc;
    r += 0.5;
  }
  
  if (aux != asfoc)
    DestroyImage(&aux);

  return(asfoc);
}

// this is the same as AsfCO due to idempotence

Image *AsfOCO(Image *img, AdjRel *A)
{
  Image *open=NULL,*close=NULL;

  open  = Open(img,A);
  close = Close(open,A);
  DestroyImage(&open);
  open  = Open(close,A);
  DestroyImage(&close);

  return(open);
}

Image *AsfnOCO(Image *img, int ntimes)
{
  Image *asfoco=NULL,*aux=NULL; 
  AdjRel *A;
  int i;
  float r=1.0;
  
  aux = ift_CopyImage(img);
  for (i=1; i <= ntimes; i++){    
    A = Circular(r);
    asfoco = AsfOCO(aux,A);
    DestroyAdjRel(&A);
    DestroyImage(&aux);
    aux = asfoco;
    r += 0.5;
  }
  
  if (aux != asfoco)
    DestroyImage(&aux);

  return(asfoco);
}

Image *AsfCO(Image *img, AdjRel *A)
{
  Image *open=NULL,*close=NULL;

  close  = Close(img,A);
  open   = Open(close,A);
  DestroyImage(&close);

  return(open);
}

Image *AsfnCO(Image *img, int ntimes)
{
  Image *asfco=NULL,*aux=NULL; 
  AdjRel *A;
  int i;
  float r=1.0;
  
  aux = ift_CopyImage(img);
  for (i=1; i <= ntimes; i++){    
    A = Circular(r);
    asfco = AsfCO(aux,A);
    DestroyAdjRel(&A);
    DestroyImage(&aux);
    aux = asfco;
    r += 0.5;
  }
  
  if (aux != asfco)
    DestroyImage(&aux);

  return(asfco);
}

Image *AsfCOC(Image *img, AdjRel *A)
{
  Image *open=NULL,*close=NULL;

  close  = Close(img,A);
  open   = Open(close,A);
  DestroyImage(&close);
  close  = Close(open,A);
  DestroyImage(&open);

  return(close);
}

Image *AsfnCOC(Image *img, int ntimes)
{
  Image *asfcoc=NULL,*aux=NULL; 
  AdjRel *A;
  int i;
  float r=1.0;
  
  aux = ift_CopyImage(img);
  for (i=1; i <= ntimes; i++){    
    A = Circular(r);
    asfcoc = AsfCOC(aux,A);
    DestroyAdjRel(&A);
    DestroyImage(&aux);
    aux = asfcoc;
    r += 0.5;
  }
  
  if (aux != asfcoc)
    DestroyImage(&aux);

  return(asfcoc);
}

void iftBasins(AnnImg *aimg, AdjRel *A)
{
  Queue *Q=NULL;
  int i,p,q,tmp,Imax,n;
  Pixel u,v;

  n = aimg->img->ncols*aimg->img->nrows;
  Imax = MaximumValue(aimg->img);
  for (p=0; p < n; p++)
    if (aimg->cost->val[p] != INT_MAX)
      if (Imax < aimg->cost->val[p])
	Imax = aimg->cost->val[p];
  
  Q = CreateQueue(Imax+1,n);

  while (aimg->seed != NULL){
    p=RemoveSet(&(aimg->seed));
    InsertQueue(Q,aimg->cost->val[p],p);
  }

  while (!EmptyQueue(Q)) {
    p=RemoveQueue(Q);

    u.x = p%aimg->img->ncols;
    u.y = p/aimg->img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	q = v.x + aimg->img->tbrow[v.y];
	if ((aimg->cost->val[p] < aimg->cost->val[q]) || 
	    (p == aimg->pred->val[q]))
	{ 
	  tmp = MAX(aimg->cost->val[p],aimg->img->val[q]);
	  if ((tmp < aimg->cost->val[q]) || 
	      (p == aimg->pred->val[q])     ) /* tmp == cost[q] =>
                                                 label[p] should
                                                 propagate because it
                                                 is the
                                                 predecessor. */
	  {	    
	    InsertQueue(Q,tmp,q);
	    aimg->cost->val[q]  = tmp;
	    aimg->pred->val[q]  = p;
	    aimg->label->val[q] = aimg->label->val[p];
	  } 
	}
      }
    }
  }
  DestroyQueue(&Q);

}

void iftDomes(AnnImg *aimg, AdjRel *A)
{
  Queue *Q=NULL;
  int i,p,q,tmp,Imax,n;
  Pixel u,v;

  n = aimg->img->ncols*aimg->img->nrows;
  Imax = MaximumValue(aimg->img);
  for (p=0; p < n; p++)
    if (aimg->cost->val[p] != INT_MAX){
      if (Imax < aimg->cost->val[p])
	Imax = aimg->cost->val[p];
    } else {
      aimg->cost->val[p] = INT_MIN;
    }


  Q = CreateQueue(Imax+1,n);

  while (aimg->seed != NULL){
    p=RemoveSet(&(aimg->seed));
    InsertQueue(Q,Imax-aimg->cost->val[p],p);
  }

  while (!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    u.x = p%aimg->img->ncols;
    u.y = p/aimg->img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	q = v.x + aimg->img->tbrow[v.y];
	if ((aimg->cost->val[p] > aimg->cost->val[q])||
	    (p == aimg->pred->val[q])){ 
	  tmp = MIN(aimg->cost->val[p],aimg->img->val[q]);
	  if ((tmp > aimg->cost->val[q]) || 
	      (p == aimg->pred->val[q])     ) /* tmp == cost[q] =>
                                                 label[p] should
                                                 propagate because it
                                                 is the
                                                 predecessor. */
	  {
	    InsertQueue(Q,Imax-tmp,q);
	    aimg->cost->val[q]  = tmp;
	    aimg->pred->val[q]  = p;
	    aimg->label->val[q] = aimg->label->val[p];
	  } 
	}
      }
    }
  }
  
  DestroyQueue(&Q);

}
  
Image *HClose(Image *img, Image *seed, AdjRel *A, int H)
{
  Image *cost=NULL;
  Queue *Q=NULL;
  int i,p,q,Imax,n;
  Pixel u,v;
 
  Imax = MaximumValue(img);
  n    = img->ncols*img->nrows;
  cost = ift_CopyImage(img);
  Q    = CreateQueue(Imax+1,n);

  for (p=0; p < n; p++){
    if (seed->val[p] > 0) {
      cost->val[p] = MIN(cost->val[p]+H,Imax);
      InsertQueue(Q,cost->val[p],p);
    } 
  }

  while (!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (cost->val[p] > cost->val[q])
	  {
	    InsertQueue(Q,cost->val[p],q);
	    cost->val[q]  = cost->val[p];
	  }
      }
    }
  }

  DestroyQueue(&Q);
  return(cost);
}


Image *HOpen(Image *img, Image *seed, AdjRel *A, int H)
{
  Image *cost=NULL;
  Queue *Q=NULL;
  int i,p,q,Imax,n;
  Pixel u,v;
 
  Imax = MaximumValue(img);
  n    = img->ncols*img->nrows;
  cost = ift_CopyImage(img);
  Q    = CreateQueue(Imax+1,n);
  for (p=0; p < n; p++){
    if (seed->val[p] > 0) {
      cost->val[p] = MAX(img->val[p]-H,0);
      InsertQueue(Q,Imax - cost->val[p],p);
    }
  }

  while (!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (cost->val[p] < cost->val[q])
	  {
	    InsertQueue(Q,Imax - cost->val[p],q);
	    cost->val[q]  = cost->val[p];
	  }
	}
      }
    }
  

  DestroyQueue(&Q);

  return(cost);
}

/*Image *iftSupRec(AnnImg *aimg, AdjRel *A)
{
  iftBasins(aimg, A);
  return(ift_CopyImage(aimg->cost));
}*/


Image *iftInfRec(AnnImg *aimg, AdjRel *A)
{
  iftDomes(aimg, A);
  return(ift_CopyImage(aimg->cost));
}

Image *SupRec(Image *img, Image *marker, AdjRel *A)
{
  Image *fcost=NULL,*cost=NULL,*fimg=NULL;
  Queue *Q=NULL;
  int i,sz,p,q,tmp,n;
  AdjPxl *N=NULL;

  sz  = FrameSize(A);
  fimg = AddFrame(img,sz,INT_MIN);
  fcost = AddFrame(marker,sz,INT_MIN);
  N  = AdjPixels(fcost,A);
  n = fcost->ncols*fcost->nrows;
  Q = CreateQueue(MaximumValue(marker)+1,n);
  
  for(p = 0; p < n; p++)
    if (fcost->val[p] != INT_MIN){
	InsertQueue(Q,fcost->val[p],p);
    }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] < fcost->val[q]){
	tmp = MAX(fcost->val[p],fimg->val[q]);
	if (tmp < fcost->val[q]){
	  UpdateQueue(Q,q,fcost->val[q],tmp);
	  fcost->val[q] = tmp;
	}
      }
    }
  }
  
  cost = RemFrame(fcost,sz);

  DestroyQueue(&Q);
  DestroyImage(&fimg);
  DestroyImage(&fcost);
  DestroyAdjPxl(&N);
  
  return(cost);
}

Image *InfRec(Image *img, Image *marker, AdjRel *A)
{
  Image *fcost=NULL,*cost=NULL,*fimg=NULL;
  GQueue *Q=NULL;
  int i,sz,p,q,tmp,n,Imax;
  AdjPxl *N=NULL;

  sz    = FrameSize(A);
  fimg  = AddFrame(img,sz,INT_MAX);
  fcost = AddFrame(marker,sz,INT_MAX);
  N     = AdjPixels(fcost,A);
  n     = fcost->ncols*fcost->nrows;
  Imax  = MaximumValue(img);
  Q     = CreateGQueue(Imax+1,n,fcost->val);
  
  SetRemovalPolicy(Q,MAXVALUE);

  for(p = 0; p < n; p++)
    if (fcost->val[p] != INT_MAX){
      InsertGQueue(&Q,p);
    }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] > fcost->val[q]){
	tmp = MIN(fcost->val[p],fimg->val[q]);
	if (tmp > fcost->val[q]){
	  UpdateGQueue(&Q,q,tmp);
	  fcost->val[q] = tmp;
	}
      }
    }
  }
  
  cost  = RemFrame(fcost,sz);

  DestroyGQueue(&Q);
  DestroyImage(&fimg);
  DestroyImage(&fcost);
  DestroyAdjPxl(&N);
  
  return(cost);
}

Image *SupRecHeap(Image *img, Image *marker, AdjRel *A)
{
  Image *fcost=NULL,*cost=NULL,*fimg=NULL;
  Heap *H=NULL;
  int i,sz,p,q,tmp,n;
  AdjPxl *N=NULL;

  sz  = FrameSize(A);
  fimg = AddFrame(img,sz,INT_MIN);
  fcost = AddFrame(marker,sz,INT_MIN);
  N  = AdjPixels(fcost,A);
  n = fcost->ncols*fcost->nrows;
  H = CreateHeap(n,fcost->val);
  
  for(i=0; i < n; i++)
    if (fcost->val[i]!=INT_MIN)
      InsertHeap(H,i);

  while(!HeapIsEmpty(H)) {
    RemoveHeap(H,&p);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (H->color[q] == GRAY){
	tmp = MAX(fcost->val[p],fimg->val[q]);
	if (tmp < fcost->val[q]){
	  fcost->val[q] = tmp;
	  GoUpHeap(H,H->pos[q]);
	}
      }
    }
  }
  
  cost = RemFrame(fcost,sz);

  DestroyHeap(&H);
  DestroyImage(&fimg);
  DestroyImage(&fcost);
  DestroyAdjPxl(&N);

  return(cost);
}

Image *RemDomes(Image *img)
{
  AdjRel *A=NULL;
  Image *marker=NULL,*oimg=NULL;
  int Imin,i,j,x,y;
  
  Imin  = MinimumValue(img);
  marker  = CreateImage(img->ncols,img->nrows);
  SetImage(marker,MAX(Imin-1,0));
  for (y=0; y < marker->nrows; y++) {
    i = marker->tbrow[y]; j = marker->ncols-1+marker->tbrow[y];
    marker->val[i] = img->val[i];
    marker->val[j] = img->val[j];
  }
  for (x=0; x < marker->ncols; x++) {
    i = x+marker->tbrow[0]; j = x+marker->tbrow[marker->nrows-1]; 
    marker->val[i] = img->val[i];
    marker->val[j] = img->val[j];
  }
  A     = Circular(1.0);
  oimg  = InfRec(img,marker,A);
  
  DestroyImage(&marker);
  DestroyAdjRel(&A);

  return(oimg);
}

Image *CloseHoles(Image *img)
{
  AdjRel *A=NULL;
  Image *marker=NULL,*cimg=NULL;
  int x,y,i,j,Imax;
  
  Imax   = MaximumValue(img);
  marker   = CreateImage(img->ncols,img->nrows);
  SetImage(marker,Imax+1);
  for (y=0; y < marker->nrows; y++) {
    i = marker->tbrow[y]; j = marker->ncols-1+marker->tbrow[y];
    marker->val[i] = img->val[i];
    marker->val[j] = img->val[j];
  }
  for (x=0; x < marker->ncols; x++) {
    i = x+marker->tbrow[0]; j = x+marker->tbrow[marker->nrows-1]; 
    marker->val[i] = img->val[i];
    marker->val[j] = img->val[j];
  }
  A      = Circular(1.0);
  cimg   = SupRec(img,marker,A);  
  DestroyImage(&marker);
  DestroyAdjRel(&A);

  return(cimg);
}

Image *OpenRec(Image *img, AdjRel *A)
{
  Image *open=NULL,*orec=NULL;
  AdjRel *A4;

  open = Open(img,A);
  A4   = Circular(1.0);
  orec = InfRec(img,open,A4);
  DestroyImage(&open);
  DestroyAdjRel(&A4);

  return(orec);
}

Image *CloseRec(Image *img, AdjRel *A)
{
  Image *close=NULL,*crec=NULL;
  AdjRel *A4;

  close = Close(img,A);
  A4    = Circular(1.0);
  crec  = SupRec(img,close,A4);
  DestroyImage(&close);
  DestroyAdjRel(&A4);

  return(crec);
}

Image *Leveling(Image *img1, Image *img2)
{
  AdjRel *A=NULL;
  Image *dil,*ero,*and,*or,*infrec,*suprec;

  A      = Circular(1.0);
  dil    = Dilate(img2,A);
  and    = And(img1,dil);
  infrec = InfRec(img1,and,A); /* and <= infrec <= img1 */

  ero    = Erode(img1,A);
  or     = Or(ero,infrec);    
  suprec = SupRec(infrec,or,A);  /* and <= suprec <= or */ 

  DestroyImage(&dil);
  DestroyImage(&ero);
  DestroyImage(&and);
  DestroyImage(&or);
  DestroyImage(&infrec);
  DestroyAdjRel(&A);

  return(suprec);
}

Image *AsfOCRec(Image *img, AdjRel *A)
{
  Image *open=NULL,*close=NULL;

  open  = OpenRec(img,A);
  close = CloseRec(open,A);
  DestroyImage(&open);

  return(close);
}


Image *AsfnOCRec(Image *img, int ntimes)
{
  Image *asfoc=NULL,*aux=NULL; 
  AdjRel *A;
  int i;
  float r=1.0;
  
  aux = ift_CopyImage(img);
  for (i=1; i <= ntimes; i++){    
    A = Circular(r);
    asfoc = AsfOCRec(aux,A);
    DestroyAdjRel(&A);
    DestroyImage(&aux);
    aux = asfoc;
    r += 0.5;
  }
  
  if (aux != asfoc)
    DestroyImage(&aux);

  return(asfoc);
}

Image *AsfOCORec(Image *img, AdjRel *A)
{
  Image *open=NULL,*close=NULL;

  open  = OpenRec(img,A);
  close = CloseRec(open,A);
  DestroyImage(&open);
  open  = OpenRec(close,A);
  DestroyImage(&close);

  return(open);
}

Image *AsfnOCORec(Image *img, int ntimes)
{
  Image *asfoco=NULL,*aux=NULL; 
  AdjRel *A;
  int i;
  float r=1.0;
  
  aux = ift_CopyImage(img);
  for (i=1; i <= ntimes; i++){    
    A = Circular(r);
    asfoco = AsfOCORec(aux,A);
    DestroyAdjRel(&A);
    DestroyImage(&aux);
    aux = asfoco;
    r += 0.5;
  }
  
  if (aux != asfoco)
    DestroyImage(&aux);

  return(asfoco);
}

Image *AsfCORec(Image *img, AdjRel *A)
{
  Image *open=NULL,*close=NULL;

  close  = CloseRec(img,A);
  open   = OpenRec(close,A);
  DestroyImage(&close);

  return(open);
}

Image *AsfnCORec(Image *img, int ntimes)
{
  Image *asfco=NULL,*aux=NULL; 
  AdjRel *A;
  int i;
  float r=1.0;
  
  aux = ift_CopyImage(img);
  for (i=1; i <= ntimes; i++){    
    A = Circular(r);
    asfco = AsfCORec(aux,A);
    DestroyAdjRel(&A);
    DestroyImage(&aux);
    aux = asfco;
    r += 0.5;
  }
  
  if (aux != asfco)
    DestroyImage(&aux);

  return(asfco);
}

Image *AsfCOCRec(Image *img, AdjRel *A)
{
  Image *open=NULL,*close=NULL;

  close  = CloseRec(img,A);
  open   = OpenRec(close,A);
  DestroyImage(&close);
  close  = CloseRec(open,A);
  DestroyImage(&open);

  return(close);
}

Image *AsfnCOCRec(Image *img, int ntimes)
{
  Image *asfcoc=NULL,*aux=NULL; 
  AdjRel *A;
  int i;
  float r=1.0;
  
  aux = ift_CopyImage(img);
  for (i=1; i <= ntimes; i++){    
    A = Circular(r);
    asfcoc = AsfCOCRec(aux,A);
    DestroyAdjRel(&A);
    DestroyImage(&aux);
    aux = asfcoc;
    r += 0.5;
  }
  
  if (aux != asfcoc)
    DestroyImage(&aux);

  return(asfcoc);
}

// As sementes da IFT local serao os pixels dos platos. Veja
// comentarios em AreaOpen.

Image *AreaClose(Image *img, int thres)
{
  Image  *area=NULL,*cost=NULL,*level=NULL,*pred=NULL,*root=NULL;           
  AdjRel *A=NULL;
  GQueue  *Q=NULL;
  int i,p,q,r=0,s,n,Imax,tmp;
  Pixel u,v;

  A        = Circular(1.0);
  Imax     = MaximumValue(img);
  area     = CreateImage(img->ncols,img->nrows);
  pred     = CreateImage(img->ncols,img->nrows);
  root     = CreateImage(img->ncols,img->nrows);
  level    = ift_CopyImage(img);
  cost     = CreateImage(img->ncols,img->nrows);
  n        = img->ncols*img->nrows;
  Q        = CreateGQueue(Imax+1,n,cost->val);

  for (p=0; p < n; p++){
    pred->val[p]  = NIL;
    root->val[p]  = p;
    cost->val[p]  = img->val[p];
    InsertGQueue(&Q,p);
  }
  
  /* Find level for local superior reconstruction */

  while (!EmptyGQueue(Q)){
    p=RemoveGQueue(Q);
      
    /* Find and update root pixel, level and area */    

    r  = SeedComp(root,p);

    if ((area->val[r]<= thres)&&(level->val[r] < img->val[p]))
      level->val[r] = img->val[p];

    area->val[r] = area->val[r] + 1;

    /* Visit the adjacent pixels */

    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (cost->val[p] < cost->val[q]){
	  root->val[q] = root->val[p];
	  pred->val[q] = p;	    	    
	}else { /* Merge two basins */
	  if (Q->L.elem[q].color == BLACK){
	    s = SeedComp(root,q);	  
	    if (r != s) {	
	      if ((area->val[s] <= thres)&&(level->val[s]< img->val[p]))
		level->val[s] = img->val[p];
	      
	      if (area->val[r] < area->val[s]){
		tmp = r;
		r   = s;
		s   = tmp;
	      }    
	      root->val[s] = r;	    
	      area->val[r] += area->val[s];
	    }
	  }
	}
      }
    }
  }
  
  /* Compute local superior reconstruction */
  
  SetRemovalPolicy(Q,MAXVALUE);

  for (p=0; p < n; p++) {
    cost->val[p]=level->val[p];
    if (pred->val[p]==NIL){
      InsertGQueue(&Q,p);
    }      
  }


  while (!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);

    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (cost->val[p] > cost->val[q])
	  {
	    if (Q->L.elem[q].color == GRAY)
	      RemoveGQueueElem(Q,q);
	    cost->val[q]  = cost->val[p];	      
	    InsertGQueue(&Q,q);
	  }
      }		  
    }
  }
  
  DestroyGQueue(&Q);
  DestroyAdjRel(&A);
  DestroyImage(&area);    
  DestroyImage(&level);    
  DestroyImage(&pred);    
  DestroyImage(&root);    

  return(cost);
}

/*  Area open com 2 IFTs: IFT-dual-watershed classica + reconstrucao local */
/*  A funcao de custo da primeira IFT eh: */
/*  J(t)   = I(t)+1 para todo t para evitar valores negativos. */
/*  f(<t>) = J(t)-1 se P(t)!=nil e J(t) se P(t)=nil (raiz) */
/*  f(pi_s . <s.t>) = min (f(pi_s,J(t)) */
/*  A primeira IFT usa as raizes (1 por maximo) como sementes e calcula o nivel de corte do domo correspondente maximizando f.   */
/*  Na segunda IFT temos a minimizacao de f: */

/*  f(<t>)=level(t) se t eh semente e I(t) no caso contrario,  */
/*                  onde level(t) <= I(t) */

/*  f(pi_s.<s,t>)=level(s) se f(pi_s) < I(t) e infinito no cc			 */

/* Image *AreaOpen(Image *img, int thres) */
/* { */
/*   Image  *area=NULL,*cost=NULL,*level=NULL,*pred=NULL,*root=NULL;            */
/*   AdjRel *A=NULL; */
/*   GQueue  *Q=NULL; */
/*   int i,p,q,r=0,s,n,Imax,tmp; */
/*   Pixel u,v; */
/*   Image *imgaux; */

/*   A        = Circular(1.0); */
/*   Imax     = MaximumValue(img); */
/*   area     = CreateImage(img->ncols,img->nrows); */
/*   pred     = CreateImage(img->ncols,img->nrows); */
/*   root     = CreateImage(img->ncols,img->nrows); */
/*   level    = ift_CopyImage(img); */
/*   cost     = CreateImage(img->ncols,img->nrows); */
/*   imgaux   = CreateImage(img->ncols,img->nrows); */
/*   n        = img->ncols*img->nrows; */
/*   Q        = CreateGQueue(Imax+3,n,cost->val); */

/*   SetRemovalPolicy(Q,MAXVALUE); */

/*   for (p=0; p < n; p++){ */
/*     imgaux->val[p]  = img->val[p]+1; */
/*     pred->val[p]    = NIL; */
/*     root->val[p]    = p; */
/*     cost->val[p]    = imgaux->val[p]-1; */
/*     InsertGQueue(&Q,p); */
/*   } */
  
/*   /\* Find level for local superior reconstruction using IFT *\/ */

/*   while (!EmptyGQueue(Q)){ */
/*     p=RemoveGQueue(Q); */
      
/*     if (pred->val[p]==NIL) // p is root */
/*       cost->val[p]=imgaux->val[p]+1; */

/*     /\* Find and update root pixel, level and area *\/     */

/*     r  = SeedComp(root,p); */

/*     if ((area->val[r]<= thres)&&(level->val[r] > img->val[p])) */
/*       level->val[r] = img->val[p]; */

/*     area->val[r] = area->val[r] + 1; */

/*     /\* Visit the adjacent pixels *\/ */

/*     u.x = p%img->ncols; */
/*     u.y = p/img->ncols; */
/*     for (i=1; i < A->n; i++){ */
/*       v.x = u.x + A->dx[i]; */
/*       v.y = u.y + A->dy[i]; */
/*       if (ValidPixel(img,v.x,v.y)){ */
/* 	q = v.x + img->tbrow[v.y]; */
/* 	if (cost->val[p] > cost->val[q]){ */
/* 	  tmp = MIN(cost->val[p],imgaux->val[q]); */
/* 	  if (tmp > cost->val[q]){ */
/* 	    RemoveGQueueElem(Q,q); */
/* 	    root->val[q] = root->val[p]; */
/* 	    pred->val[q] = p; */
/* 	    cost->val[q] = tmp; */
/* 	    InsertGQueue(&Q,q); */
/* 	 }  */
/* 	}else { /\* Merge two basins *\/ */
/* 	  if (Q->L.elem[q].color == BLACK){ */
/* 	    s  = SeedComp(root,q);	   */

/* 	    if (r != s) {	 */
/* 	      if ((area->val[s] <= thres)&&(level->val[s] > img->val[p])) */
/* 		level->val[s] = img->val[p]; */
	      
/* 	      if (area->val[r] < area->val[s]){ */
/* 		tmp = r; */
/* 		r   = s; */
/* 		s   = tmp; */
/* 	      } */
	      
/* 	      root->val[s] = r;	     */
/* 	      area->val[r] += area->val[s]; */
/* 	    } */
/* 	  } */
/* 	} */
/*       } */
/*     } */
/*   } */

/*   /\* Compute local superior reconstruction *\/ */
  
/*   SetRemovalPolicy(Q,MINVALUE); */

/*   for (p=0; p < n; p++) { */
/*     cost->val[p]=level->val[p]; */
/*     if (pred->val[p]==NIL){ */
/*       InsertGQueue(&Q,p); */
/*     }       */
/*   } */


/*   while (!EmptyGQueue(Q)) { */
/*     p=RemoveGQueue(Q); */

/*     u.x = p%img->ncols; */
/*     u.y = p/img->ncols; */
/*     for (i=1; i < A->n; i++){ */
/*       v.x = u.x + A->dx[i]; */
/*       v.y = u.y + A->dy[i]; */
/*       if (ValidPixel(img,v.x,v.y)){ */
/* 	q = v.x + img->tbrow[v.y]; */
/* 	if (cost->val[p] < cost->val[q]) */
/* 	  { */
/* 	    if (Q->L.elem[q].color == GRAY) */
/* 	      RemoveGQueueElem(Q,q); */
/* 	    cost->val[q]  = cost->val[p];	       */
/* 	    InsertGQueue(&Q,q); */
/* 	  } */
/*       }		   */
/*     } */
/*   } */
  
/*   DestroyGQueue(&Q); */
/*   DestroyAdjRel(&A); */
/*   DestroyImage(&area);     */
/*   DestroyImage(&level);     */
/*   DestroyImage(&pred);     */
/*   DestroyImage(&root);     */
/*   DestroyImage(&imgaux);     */


/*   return(cost); */
/* } */


// Area open onde a primeira IFT eh substituida por uma funcao que
// identifica os platos como raizes de uma floresta espalhada. O
// processamento ainda deve ser ordenado (decrescente no brilho) para
// que o nivel seja calculado. As raizes da primeira funcao serao as
// sementes da IFT local. O AreaClose seguira este esquema.

Image *AreaOpen(Image *img, int thres)
{
  Image  *area=NULL,*cost=NULL,*level=NULL,*pred=NULL,*root=NULL;           
  AdjRel *A=NULL;
  GQueue  *Q=NULL;
  int i,p,q,r=0,s,n,Imax,tmp;
  Pixel u,v;

  A        = Circular(1.0);
  Imax     = MaximumValue(img);
  area     = CreateImage(img->ncols,img->nrows);
  pred     = CreateImage(img->ncols,img->nrows);
  root     = CreateImage(img->ncols,img->nrows);
  level    = ift_CopyImage(img);
  cost     = CreateImage(img->ncols,img->nrows);
  n        = img->ncols*img->nrows;
  Q        = CreateGQueue(Imax+1,n,cost->val);

  SetRemovalPolicy(Q,MAXVALUE);

  for (p=0; p < n; p++){
    pred->val[p]    = NIL;
    root->val[p]    = p;
    cost->val[p]    = img->val[p];
    InsertGQueue(&Q,p);
  }
  
  /* Find level for local superior reconstruction using IFT */

  while (!EmptyGQueue(Q)){
    p=RemoveGQueue(Q);
      
    /* Find and update root pixel, level and area */    

    r  = SeedComp(root,p);

    if ((area->val[r]<= thres)&&(level->val[r] > img->val[p]))
      level->val[r] = img->val[p];

    area->val[r] = area->val[r] + 1;

    /* Visit the adjacent pixels */

    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (cost->val[p] > cost->val[q]){
	  root->val[q] = root->val[p];
	  pred->val[q] = p;
	} else { /* Merge two basins */
	  if (Q->L.elem[q].color == BLACK){
	    s  = SeedComp(root,q);	  

	    if (r != s) {	
	      if ((area->val[s] <= thres)&&(level->val[s] > img->val[p]))
		level->val[s] = img->val[p];
	      
	      if (area->val[r] < area->val[s]){
		tmp = r;
		r   = s;
		s   = tmp;
	      }
	      
	      root->val[s] = r;	    
	      area->val[r] += area->val[s];
	    }
	  }
	}
      }
    }
  }

  /* Compute local superior reconstruction */
  
  SetRemovalPolicy(Q,MINVALUE);

  for (p=0; p < n; p++) {
    cost->val[p]=level->val[p];
    if (pred->val[p]==NIL){
      InsertGQueue(&Q,p);
    }      
  }


  while (!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);

    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (cost->val[p] < cost->val[q])
	  {
	    if (Q->L.elem[q].color == GRAY)
	      RemoveGQueueElem(Q,q);
	    cost->val[q]  = cost->val[p];	      
	    InsertGQueue(&Q,q);
	  }
      }		  
    }
  }
  
  DestroyGQueue(&Q);
  DestroyAdjRel(&A);
  DestroyImage(&area);    
  DestroyImage(&level);    
  DestroyImage(&pred);    
  DestroyImage(&root);    


  return(cost);
}


/* It assumes that the next operation is a erosion, but it may be
   another dilation if you remove comments below. */

Image *DilateBin(Image *bin, Set **seed, float radius)
{
  Image *cost=NULL,*root=NULL,*ero=NULL,*boundr=NULL;
  Queue *Q=NULL;
  int i,p,q,n,sz;
  Pixel u,v,w;
  int *sq=NULL,tmp=INT_MAX,dx,dy;
  float dist;
  AdjRel *A=NULL;
  Image *dil=NULL;

  /* Compute seeds */
  
  if (*seed == NULL) {
    A      = Circular(1.0);
    ero    = Erode(bin,A);
    boundr = Diff(bin,ero);
    DestroyImage(&ero);
    DestroyAdjRel(&A);
    n    = boundr->ncols*boundr->nrows;
    for (p=0; p < n; p++)
      if (boundr->val[p]==1){
	InsertSet(seed,p);
      }
    DestroyImage(&boundr);    
  }

  /* Dilate image */

  dil  = ift_CopyImage(bin);  
  dist = (radius*radius);
  A  = Circular(1.5);
  n  = MAX(dil->ncols,dil->nrows);
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  cost = CreateImage(dil->ncols,dil->nrows);
  root = CreateImage(dil->ncols,dil->nrows);
  n    = dil->ncols*dil->nrows;
  sz   = FrameSize(A);
  Q    = CreateQueue(2*sz*(sz+dil->ncols+dil->nrows),n);
  SetImage(cost,INT_MAX);

  while (*seed != NULL){
    p=RemoveSet(seed);
    cost->val[p]=0;
    root->val[p]=p;
    InsertQueue(Q,cost->val[p]%Q->C.nbuckets,p);
  }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (cost->val[p] <= dist){

      dil->val[p] = 1;

      /* Seeds for dilation, if we wanted to compute a sequence of dilations 

      if (((sq[Dx->val[p]+1]+sq[Dy->val[p]]) > dist)||
	  ((sq[Dx->val[p]]+sq[Dy->val[p]+1]) > dist)){
	InsertSet(seed,p);
      }

      */

      u.x = p%dil->ncols;
      u.y = p/dil->ncols;
      w.x = root->val[p]%dil->ncols;
      w.y = root->val[p]/dil->ncols;

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(dil,v.x,v.y)){
	  q = v.x + dil->tbrow[v.y];
	  if ((cost->val[p] < cost->val[q])&&(dil->val[q]==0)){	   
	    dx  = abs(v.x-w.x);
	    dy  = abs(v.y-w.y);
	    tmp = sq[dx] + sq[dy];
	    if (tmp < cost->val[q]){
	      if (cost->val[q] == INT_MAX)
		InsertQueue(Q,tmp%Q->C.nbuckets,q);
	      else
		UpdateQueue(Q,q,cost->val[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	      cost->val[q]  = tmp;
	      root->val[q]  = root->val[p];
	    }
	  }
	}
      }
    } else { /* Seeds for erosion */
      InsertSet(seed,p);
    }
  }

  free(sq);
  DestroyQueue(&Q);
  DestroyImage(&root);
  DestroyImage(&cost);
  DestroyAdjRel(&A);

  return(dil);
}

/* It assumes that the next operation is a erosion, but it may be
   another dilation if you remove comments below. */

Image *FastDilate(Image *img, float radius)
{
  Image *dil,*pred;
  Queue *Q;
  int *FIFO=NULL;
  int first=0,last=0;
  int Imax,p,q,r,i,n,dx,dy;
  int *sq;
  float r2;
  AdjRel *A;
  char *C;
  Pixel u,v,w;

  Imax = MaximumValue(img);
  n    = img->ncols*img->nrows;
  Q    = CreateQueue(Imax+1,n);
  A    = Circular(1.5);
  r2   = radius*radius;
  C    = AllocCharArray(n);
  sq   = AllocIntArray(p=MAX(img->ncols,img->nrows));
  for(i=0; i < p; i++)
    sq[i]=i*i;

  dil   = ift_CopyImage(img);
  pred  = CreateImage(img->ncols,img->nrows);
  for(p=0; p < n; p++) {
    pred->val[p]=p;
    if (img->val[p] > 0)
      InsertQueue(Q,Imax-img->val[p],p);
  }

  FIFO  = AllocIntArray(n);
  while(!EmptyQueue(Q)){
    r     = RemoveQueue(Q);
    w.x   = r%img->ncols; 
    w.y   = r/img->ncols; 
    Imax  = img->val[r];
    /*    
    p     = r;
    while(pred->val[p]!=p){
      u.x   = p%img->ncols; 
      u.y   = p/img->ncols; 
      dx = abs(u.x-w.x); 
      dy = abs(u.y-w.y);
      if (sq[dx]+sq[dy] <= r2){
	if (img->val[p] > Imax)
	  Imax = img->val[p];
      } else 
	break;
      p = pred->val[p];
    }
    */
    first = last = 0;
    memset(C,0,n);
    FIFO[last]=r;      
    last++;  
    C[r]=GRAY;
    while(first != last){
      p     = FIFO[first];
      u.x   = p%img->ncols; 
      u.y   = p/img->ncols; 
      first++;
      C[p]  = BLACK;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(img,v.x,v.y)){
	  q  = v.x + img->tbrow[v.y];
	  if(Imax > img->val[q]){
	    dx = abs(v.x-w.x); 
	    dy = abs(v.y-w.y);
	    pred->val[q]=p;
	    if (sq[dx]+sq[dy] <= r2){
	      if (Imax > dil->val[q]){
		dil->val[q]=Imax;
	      }
	      if (C[q]==WHITE){
		FIFO[last]=q;      
		C[q]=GRAY;
		last++;     
	      }
	    }
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);
  DestroyQueue(&Q);
  DestroyImage(&pred);
  free(sq);
  free(C);
  return(dil);
}

/* It assumes that the next operation is a dilation, but it may
   be an erosion if you remove comments below. */

Image *ErodeBin(Image *bin, Set **seed, float radius)
{
  Image *ero=NULL,*boundr=NULL,*dil=NULL;
  Pixel u,v,w;
  Image *cost=NULL,*root;
  Queue *Q=NULL;
  int i,p,q,n,sz;
  int *sq=NULL,tmp=INT_MAX,dx,dy;
  AdjRel *A=NULL;
  float dist;

  /* Compute seeds */
  
  if (*seed == NULL) {
    A      = Circular(1.0);
    dil    = Dilate(bin,A);
    boundr = Diff(dil,bin);
    DestroyImage(&dil);
    DestroyAdjRel(&A);
    n    = boundr->ncols*boundr->nrows;
    for (p=0; p < n; p++)
      if (boundr->val[p]==1){
	InsertSet(seed,p);
      }
    DestroyImage(&boundr);    
  }

  /* Erode image */

  ero  = ift_CopyImage(bin);  
  dist = (radius*radius);
  A  = Circular(1.5);
  n  = MAX(ero->ncols,ero->nrows);
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  cost = CreateImage(ero->ncols,ero->nrows);
  root = CreateImage(ero->ncols,ero->nrows);
  SetImage(cost,INT_MAX);
  n    = ero->ncols*ero->nrows;
  sz   = FrameSize(A);
  Q    = CreateQueue(2*sz*(sz+ero->ncols+ero->nrows),n);
  
  while (*seed != NULL){
    p=RemoveSet(seed);
    cost->val[p]=0;
    root->val[p]=p;
    InsertQueue(Q,cost->val[p]%Q->C.nbuckets,p);
  }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (cost->val[p] <= dist){

      ero->val[p] = 0;

      /* Seeds for erosion if we wanted to compute sequences of erosions

      if (((sq[Dx->val[p]+1]+sq[Dy->val[p]]) > dist)||
	  ((sq[Dx->val[p]]+sq[Dy->val[p]+1]) > dist)){
	InsertSet(seed,p);
      }

      */

      u.x = p%ero->ncols;
      u.y = p/ero->ncols;
      w.x = root->val[p]%ero->ncols;
      w.y = root->val[p]/ero->ncols;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(ero,v.x,v.y)){
	  q = v.x + ero->tbrow[v.y];
	  if ((cost->val[p] < cost->val[q])&&(ero->val[q]==1)){
	    dx  = abs(v.x-w.x);
	    dy  = abs(v.y-w.y);
	    tmp = sq[dx] + sq[dy];
	    if (tmp < cost->val[q]){
	      if (cost->val[q] == INT_MAX)
		InsertQueue(Q,tmp%Q->C.nbuckets,q);
	      else
		UpdateQueue(Q,q,cost->val[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	      cost->val[q] = tmp;
	      root->val[q] = root->val[p];
	    }
	  }
	}
      }
    } else {  /* Seeds for dilation */
      InsertSet(seed,p);
    }
  }
  
  free(sq);
  DestroyQueue(&Q);
  DestroyImage(&root);
  DestroyImage(&cost);
  DestroyAdjRel(&A);

  return(ero);
}

Image *CloseBin(Image *bin, float radius)
{
  Image *close=NULL,*dil=NULL;
  Set *seed=NULL;

  dil   = DilateBin(bin,&seed,radius);
  close = ErodeBin(dil,&seed,radius);
  DestroyImage(&dil);
  DestroySet(&seed);

  return(close);
}

Image *OpenBin(Image *bin, float radius)
{
  Image *open=NULL,*ero=NULL;
  Set *seed=NULL;

  ero   = ErodeBin(bin,&seed,radius);
  open  = DilateBin(ero,&seed,radius);
  DestroyImage(&ero);
  DestroySet(&seed);

  return(open);
}

Image *AsfOCBin(Image *bin, float radius)
{
  Image *dil=NULL,*ero=NULL;
  Set *seed=NULL;

  ero = ErodeBin(bin,&seed,radius);
  dil = DilateBin(ero,&seed,2.0*radius);
  DestroyImage(&ero);
  ero = ErodeBin(dil,&seed,radius);
  DestroyImage(&dil);
  DestroySet(&seed);

  return(ero);
}



static __inline__ int findRoot(int *parent, int x) {
    if (parent[x] >= 0) {
        parent[x] = findRoot(parent, parent[x]);
        return parent[x];
    }
    return x;
}

/* ordena o vetor de Pixel pelo valor x. */
static __inline__ void radixSort (int byte, int N, Pixel *source, Pixel *dest) {
  int count[256];
  int index[256];
  int i;
  memset (count, 0, sizeof (count));
  for (i = 0; i < N; i++) {
    count[(source[i].x >> (byte * 8)) & 0xff]++;
  }

  index[0] = 0;
  for (i = 1; i < 256; i++ ) {
    index[i] = index[i - 1] + count[i - 1];
  }
  for (i = 0; i < N; i++ ) {
    dest[index[(source[i].x >> (byte * 8)) & 0xff]++] = source[i];
  }
}

/* area open com union-find. */
Image *FastAreaOpen(Image *img, int thres) {

  int i, p, q, ip, iq, n, x, y;
  int *sortedPixel, *area, *parent;
  Pixel *pixArray, *pixArray2;
  AdjRel *A;
  Image *areaOpen;



  A = Circular(1.0);
  n = img->ncols * img->nrows;
  
  area = calloc(n, sizeof(int));
  parent = calloc(n, sizeof(int));
  
  /* ordena os valores dos pixels por RadixSort */
  pixArray = (Pixel *) malloc(n * sizeof(Pixel));
  pixArray2 = (Pixel *) malloc(n * sizeof(Pixel));
  for (p = 0; p < n; p++) {
    pixArray[p].x = img->val[p]; // x = brilho do pixel
    pixArray[p].y = p; // y = indice do pixel
  }
  sortedPixel = malloc(n * sizeof(int));
  /* se a imagem for 8 bits, chama radixSort somente uma vez. */
  if (MaximumValue(img) <= 255) {
    radixSort(0, n, pixArray, pixArray2);
    /* guarda em sortedPixel os indices dos pixels ordenados em ordem decrescente. */
    for (p = n - 1; p >= 0; p--) {
      sortedPixel[p] = pixArray2[n - p - 1].y;
    }
  } else {
    radixSort(0, n, pixArray, pixArray2);
    radixSort(1, n, pixArray2, pixArray);
    /* guarda em sortedPixel os indices dos pixels ordenados em ordem decrescente. */
    for (p = n - 1; p >= 0; p--) {
      sortedPixel[p] = pixArray[n - p - 1].y;
    }
  }
  free(pixArray);
  free(pixArray2);

#define ACTIVE -1
#define INACTIVE -2


  /* processa os pixels em ordem de brilho decrescente. */

  parent[sortedPixel[0]] = ACTIVE;
  area[sortedPixel[0]] = 1;
  for (p = 1; p < n; p++) {
    ip = sortedPixel[p];
    if (img->val[ip] != img->val[sortedPixel[p - 1]]) {
      q = p - 1;
      while ((q >= 0) && (img->val[sortedPixel[q]] == img->val[sortedPixel[p - 1]])) {
	iq = sortedPixel[q];
	if ((parent[iq] == ACTIVE) && (area[iq] > thres)) {
	  parent[iq] = INACTIVE;
	  area[iq] = 1;
	}
	q--;
      }
    }
    parent[ip] = ACTIVE;
    area[ip] = 1;
    
    /* Processa os vizinhos de ip ja visitados. */
    for (i = 1; i < A->n; i++) {
      x = (ip % img->ncols) + A->dx[i];
      y = (ip / img->ncols) + A->dy[i];
      if (ValidPixel(img, x, y)) {
	q = x + img->tbrow[y];
	/* se ja foi visitado, tenta a uniao. */
	if (area[q] != 0) {
	  
	  iq = findRoot(parent, q);
	  if (iq != ip) {
	    if ((img->val[iq] == img->val[ip]) || (parent[iq] == ACTIVE)) {

	      
	      if ((parent[ip] == ACTIVE) && (parent[iq] == ACTIVE)) {
		area[ip] += area[iq]; // merge de x e y em y
		area[iq] = 1;
	      } else if (parent[iq] == ACTIVE) {
		area[iq] = 1;
	      } else {
		area[ip] = 1;
		parent[ip] = INACTIVE;
	      }
	      parent[iq] = ip;
	      
	    } else if (parent[ip] == ACTIVE) {
	      parent[ip] = INACTIVE;
	      area[ip] = 1;
	    }
	  }

	}
      }
    }
  }

  areaOpen = CreateImage(img->ncols, img->nrows);
  for (p = n - 1; p >= 0; p--) {
    q = sortedPixel[p];
    if (parent[q] < 0) {
      areaOpen->val[q] = img->val[q];
    } else {
      areaOpen->val[q] = areaOpen->val[parent[q]];
    }
  }
  
  free(area);
  free(parent);
  free(sortedPixel);
  DestroyAdjRel(&A);


  return areaOpen;
}

/* area close com union-find. */
Image *FastAreaClose(Image *img, int thres) {

  int i, p, q, ip, iq, n, x, y;
  int *sortedPixel, *area, *parent;
  Pixel *pixArray, *pixArray2;
  AdjRel *A;
  Image *areaClose;

  A = Circular(1.0);
  n = img->ncols * img->nrows;
  
  area = calloc(n, sizeof(int));
  parent = calloc(n, sizeof(int));
  
  /* ordena os valores dos pixels por RadixSort */
  pixArray = (Pixel *) malloc(n * sizeof(Pixel));
  pixArray2 = (Pixel *) malloc(n * sizeof(Pixel));
  for (p = 0; p < n; p++) {
    pixArray[p].x = img->val[p]; // x = brilho do pixel
    pixArray[p].y = p; // y = indice do pixel
  }
  sortedPixel = malloc(n * sizeof(int));
  /* se a imagem for 8 bits, chama radixSort somente uma vez. */
  if (MaximumValue(img) <= 255) {
    radixSort(0, n, pixArray, pixArray2);
    /* guarda em sortedPixel os indices dos pixels ordenados em ordem crescente. */
    for (p = n - 1; p >= 0; p--) {
      sortedPixel[p] = pixArray2[p].y;
    }
  } else {
    radixSort(0, n, pixArray, pixArray2);
    radixSort(1, n, pixArray2, pixArray);
    /* guarda em sortedPixel os indices dos pixels ordenados em ordem crescente. */
    for (p = n - 1; p >= 0; p--) {
      sortedPixel[p] = pixArray[p].y;
    }
  }
  free(pixArray);
  free(pixArray2);

#define ACTIVE -1
#define INACTIVE -2
  
  /* processa os pixels em ordem de brilho crescente. */
  parent[sortedPixel[0]] = ACTIVE;
  area[sortedPixel[0]] = 1;
  for (p = 1; p < n; p++) {
    ip = sortedPixel[p];
    if (img->val[ip] != img->val[sortedPixel[p - 1]]) {
      q = p - 1;
      while ((q >= 0) && (img->val[sortedPixel[q]] == img->val[sortedPixel[p - 1]])) {
	iq = sortedPixel[q];
	if ((parent[iq] == ACTIVE) && (area[iq] > thres)) {
	  parent[iq] = INACTIVE;
	  area[iq] = 1;
	}
	q--;
      }
    }
    parent[ip] = ACTIVE;
    area[ip] = 1;
    
    /* Processa os vizinhos de ip ja visitados. */
    for (i = 1; i < A->n; i++) {
      x = (ip % img->ncols) + A->dx[i];
      y = (ip / img->ncols) + A->dy[i];
      if (ValidPixel(img, x, y)) {
	q = x + img->tbrow[y];
	/* se ja foi visitado, tenta a uniao. */
	if (area[q] != 0) {
	  
	  iq = findRoot(parent, q);
	  if (iq != ip) {
	    if ((img->val[iq] == img->val[ip]) || (parent[iq] == ACTIVE)) {
	      
	      if ((parent[ip] == ACTIVE) && (parent[iq] == ACTIVE)) {
		area[ip] += area[iq]; // merge de x e y em y
		area[iq] = 1;
	      } else if (parent[iq] == ACTIVE) {
		area[iq] = 1;
	      } else {
		area[ip] = 1;
		parent[ip] = INACTIVE;
	      }
	      parent[iq] = ip;
	      
	    } else if (parent[ip] == ACTIVE) {
	      parent[ip] = INACTIVE;
	      area[ip] = 1;
	    }
	  }

	}
      }
    }
  }
  
  areaClose = CreateImage(img->ncols, img->nrows);
  for (p = n - 1; p >= 0; p--) {
    q = sortedPixel[p];
    if (parent[q] < 0) {
      areaClose->val[q] = img->val[q];
    } else {
      areaClose->val[q] = areaClose->val[parent[q]];
    }
  }
  
  free(area);
  free(parent);
  free(sortedPixel);
  DestroyAdjRel(&A);

  return areaClose;
}

Image *FeatureGradient(Image *img, int nfeats, int maxval)
{
  float   dist,gx,gy;
  int     i,j,p,q,s,n=img->ncols*img->nrows,Imax;
  Pixel   u,v;
  AdjRel *A8=Circular(1.5),*A=NULL;
  float  *mg=AllocFloatArray(A8->n);
  Image  *grad=CreateImage(img->ncols,img->nrows);
  float  **feat=(float **)calloc(nfeats,sizeof(float *));
  Image *img1,*img2;

  /* Compute multiscale features with nfeats scales */

  for (i=0; i < nfeats; i++) 
    feat[i]=AllocFloatArray(n);

  Imax  = MaximumValue(img);
  img1  = ift_CopyImage(img);

  for (s=1; s <= nfeats; s=s+1) {
    A  = Circular(s);
    img2 = AsfOCRec(img1,A);
    for (i=0; i < n; i++) {
      feat[s-1][i] = (float)img2->val[i]/(float)Imax;
    }
    DestroyImage(&img2);
    DestroyAdjRel(&A);
  }
  DestroyImage(&img1);

  /* Compute Gradient */

  for (i=0; i < A8->n; i++)
    mg[i]=sqrt(A8->dx[i]*A8->dx[i]+A8->dy[i]*A8->dy[i]);

  for (u.y=0; u.y < img->nrows; u.y++)
    for (u.x=0; u.x < img->ncols; u.x++) {
      p = u.x + img->tbrow[u.y];
      gx = gy = 0.0;
      for (i=1; i < A8->n; i++) {
	v.x = u.x + A8->dx[i];
	v.y = u.y + A8->dy[i];
	if (ValidPixel(img,v.x,v.y)){
	  q = v.x + img->tbrow[v.y];
	  dist = 0;
	  for (j=0; j < nfeats; j++) 
	    dist += feat[j][q]-feat[j][p];
            gx  += dist*A8->dx[i]/mg[i];
            gy  += dist*A8->dy[i]/mg[i];
          }
      }
      //      gx = gx/nfeats; gy = gy/nfeats;
      grad->val[p]=(int)(maxval*sqrt(gx*gx + gy*gy));
    }

  free(mg);
  DestroyAdjRel(&A8);
  for (i=0; i < nfeats; i++)
    free(feat[i]);
  free(feat);
  return(grad);
}
