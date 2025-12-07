#include "genift.h"
#include "queue.h"
#include "image.h"

int FPeak(AnnImg *aimg, int p, int q)
{
  return(MAX(aimg->cost->val[p],aimg->img->val[q]));
}

int FLPeak(AnnImg *aimg, int p, int q)
{
  if (aimg->cost->val[p] > aimg->img->val[q])
    return(aimg->cost->val[p]);
  else
    return(INT_MAX);
}

int FMax(AnnImg *aimg, int p, int q)
{
  return(MAX(aimg->cost->val[p],abs(aimg->img->val[q]-aimg->img->val[p])));
}

int FIni(AnnImg *aimg, int p, int q) 
{
  if (aimg->img->val[p] <= aimg->img->val[q])
    return(aimg->cost->val[p]);
  else
    return(INT_MAX);
}

int FEucl(AnnImg *aimg, int p, int q)
{
  int dx,dy;
  Pixel u,v;
  
  u.x = (aimg->label->val[p]%aimg->img->ncols);
  u.y = (aimg->label->val[p]/aimg->img->ncols);
  v.x = q%aimg->img->ncols;
  v.y = q/aimg->img->ncols;
  dx = v.x-u.x;
  dy = v.y-u.y;
  return(dx*dx + dy*dy); 
}

int FSum(AnnImg *aimg, int p, int q)
{
  return(aimg->cost->val[p]+aimg->img->val[q]);
}

int FCham(AnnImg *aimg, int p, int q)
{
  Pixel u,v;

  u.x = p%aimg->img->ncols;
  u.y = p/aimg->img->ncols;
  v.x = q%aimg->img->ncols;
  v.y = q/aimg->img->ncols;
  if ((u.x == v.x)||(u.y == v.y)) /* 4-neighbors */
    return(aimg->cost->val[p]+17);
  else 
    return(aimg->cost->val[p]+24);
}

void IFT(AnnImg *aimg, AdjRel *A, PathCost Pcost, int maxinc)
{
  /*  Heap *H=NULL; */
  Queue *Q=NULL;
  int i,p,q,n,cost;
  Pixel u,v;

  n = aimg->img->ncols*aimg->img->nrows;
  Q = CreateQueue(maxinc+1,n);

  //  H = CreateHeap(n,aimg->cost->val);
    
  while (aimg->seed != NULL){
    p=RemoveSet(&(aimg->seed));
    //    InsertHeap(H,p);
    InsertQueue(Q,aimg->cost->val[p]%Q->C.nbuckets,p);
  }
  //  while(!HeapIsEmpty(H)) {
  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    //    RemoveHeap(H,&p);
    u.x = p%aimg->img->ncols;
    u.y = p/aimg->img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	q = v.x + aimg->img->tbrow[v.y];
	//if (H->color[q] != BLACK){
	if (aimg->cost->val[p] < aimg->cost->val[q]){
	  cost = Pcost(aimg,p,q); 
	  if (cost < aimg->cost->val[q]){
	    if (aimg->cost->val[q] == INT_MAX)
	      InsertQueue(Q,cost%Q->C.nbuckets,q);
	    else
	      UpdateQueue(Q,q,aimg->cost->val[q]%Q->C.nbuckets,cost%Q->C.nbuckets);
	    aimg->cost->val[q]  = cost;
	    aimg->pred->val[q]  = p;
	    aimg->label->val[q] = aimg->label->val[p];
	    /*	    
		    if (H->color[q] == WHITE)
		    InsertHeap(H,q);
		    else
		    GoUp(H,H->pos[q]);
	    */
	  }
	}
      }
    }
  }
  //  DestroyHeap(&H);
  DestroyQueue(&Q);
}
