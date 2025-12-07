#include "analysis.h"
#include "segmentation.h"
#include "radiometric.h"
#include "genift.h"
#include "adjacency.h"
#include "queue.h"
#include "heap.h"
#include "mathematics.h"

Image   *ObjectBorder(Image *bin)
{
  Image *border=CreateImage(bin->ncols,bin->nrows);
  int p,q,i;
  Pixel u,v;
  AdjRel *A=Circular(1.0);

  for (u.y=0; u.y < bin->nrows; u.y++) 
    for (u.x=0; u.x < bin->ncols; u.x++) {
      p = u.x + bin->tbrow[u.y];
      if (bin->val[p]>0){
	for (i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  if (ValidPixel(bin,v.x,v.y)){
	    q = v.x + bin->tbrow[v.y];
	    if (bin->val[q]==0){
	      border->val[p]=1;
	      break;
	    }
	  }
	}
      }
    }
  DestroyAdjRel(&A);
  return(border);
}

Image *Perimeter(Image *bin)
{
  int p,n;
  Image *cont,*perim;
  Curve *hist;

  cont  = LabelContour(bin);
  n     = cont->ncols*cont->nrows;
  perim = CreateImage(cont->ncols,cont->nrows);
  hist  = Histogram(cont);
  for (p=0; p < n; p++)
    if (cont->val[p] > 0)
      perim->val[p] = hist->Y[cont->val[p]];

  DestroyCurve(&hist);
  DestroyImage(&cont);

  return(perim);
}

Image *Area(Image *bin)
{
  int p,n;
  Image *label,*area;
  Curve *hist;
  AdjRel *A;

  A     = Circular(1.5);
  label = LabelBinComp(bin,A);
  n     = label->ncols*label->nrows;
  area  = CreateImage(label->ncols,label->nrows);
  hist  = Histogram(label);

  for (p=0; p < n; p++)
    if (label->val[p] > 0)
      area->val[p] = hist->Y[label->val[p]];

  DestroyCurve(&hist);
  DestroyImage(&label);
  DestroyAdjRel(&A);

  return(area);
}

Image   *GeomCenter(Image *bin)
{
  int p,i,n,Lmax;
  Image *label,*geom;
  float *X,*Y,*N;
  AdjRel *A;
  
  A     = Circular(1.5);
  label = LabelBinComp(bin,A);
  geom  = CreateImage(bin->ncols,bin->nrows);
  Lmax  = MaximumValue(label);
  X     = AllocFloatArray(Lmax+1);
  Y     = AllocFloatArray(Lmax+1);
  N     = AllocFloatArray(Lmax+1);
  n     = bin->ncols*bin->nrows;
  for (p=0; p < n; p++){
    X[label->val[p]] += (float)(p%label->ncols);
    Y[label->val[p]] += (float)(p/label->ncols);
    N[label->val[p]] ++;
  }
  for (i=0; i <= Lmax; i++){
    X[i] /= N[i];
    Y[i] /= N[i];
  }
  for (p=0; p < n; p++)
    geom->val[p] = (int)X[label->val[p]] + 
      geom->tbrow[(int)Y[label->val[p]]];

  DestroyImage(&label);
  DestroyAdjRel(&A);
  free(X);
  free(Y);
  free(N);

  return(geom);
}

Image *CompMSSkel(AnnImg *aimg)
{
  int i,p,q,n,maxd1,maxd2,d1,d2,MaxD;
  Pixel u,v;
  int sign=1,s2;
  Image *msskel,*cont=NULL,*perim=NULL,*seed=NULL;
  AdjRel *A;

  /* Compute MS Skeletons */

  cont   = LabelContour(aimg->img);
  perim  = Perimeter(aimg->img);
  seed   = CompPaths(aimg->pred);
  A      = Circular(1.0);
  n      = aimg->label->ncols*aimg->label->nrows;
  msskel = CreateImage(aimg->label->ncols,aimg->label->nrows);

  MaxD = INT_MIN;
  for (p=0; p < n; p++) {
    if (aimg->pred->val[p] != p) {  /* It eliminates the countors and
                                       already takes into account the
                                       side option */
      u.x = p%aimg->label->ncols;
      u.y = p/aimg->label->ncols;
      maxd1 = maxd2 = INT_MIN;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(aimg->label,v.x,v.y)){
	  q = v.x + aimg->label->tbrow[v.y];
	  if (cont->val[seed->val[p]] == cont->val[seed->val[q]]){ 
	    d2   = aimg->label->val[q]-aimg->label->val[p];
	    s2   = 1;
	    //	    if (d2 > (perim->val[seed->val[p]]-1-d2)){
	    if (d2 > (perim->val[seed->val[p]]-d2)){
	      s2 = -1;
	      //	      d2 = (perim->val[seed->val[p]]-1-d2);
	      d2 = (perim->val[seed->val[p]]-d2);
	    } 
	    if (d2 > maxd2){
	      maxd2 = d2;
	      sign  = s2;
	    }
	  } else {
	    d1 = cont->val[seed->val[q]] - cont->val[seed->val[p]];
	    if (d1 > maxd1) 
	      maxd1 = d1;
	  }
	}
      }
      if (maxd1 > 0) {
	msskel->val[p] = INT_MAX;
      } else {
	msskel->val[p] = sign*maxd2;
	if (msskel->val[p] > MaxD)
	  MaxD = msskel->val[p];    
      }
    }
  }

  for (p=0; p < n; p++) { /* Set up SKIZ */
    if (msskel->val[p] == INT_MAX)
      msskel->val[p] = MaxD + 1;
  }

  DestroyImage(&cont);
  DestroyImage(&perim);
  DestroyImage(&seed);
  DestroyAdjRel(&A);

  return(msskel);
}

Image *Skeleton(Image *msskel, float perc)
{
  Image *skel = NULL;
  int p ,n, thres;
  
  skel  = Abs(msskel);
  thres = (int)((MaximumValue(skel)*perc)/100.0);      
  n = skel->ncols*skel->nrows;
  for (p=0; p < n; p++)
    if (skel->val[p] >= thres) 
      skel->val[p]=1;
    else
      skel->val[p]=0;
  
  return(skel);
}

Image *CompSKIZ(AnnImg *aimg)
{
  Image *skiz;
  int i,p,q,n,maxd,d;
  AdjRel *A;
  Pixel u,v;

  A      = Circular(1.0);
  n      = aimg->label->ncols*aimg->label->nrows;
  skiz   = CreateImage(aimg->label->ncols,aimg->label->nrows);

  for (p=0; p < n; p++) {
    if (aimg->pred->val[p] != p) {
      u.x = p%aimg->label->ncols;
      u.y = p/aimg->label->ncols;
      maxd = INT_MIN;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(aimg->label,v.x,v.y)){
	  q = v.x + aimg->label->tbrow[v.y];
	  d = aimg->label->val[q]-aimg->label->val[p];
	  if (d > maxd)
	    maxd = d;
	}
      }
      if (maxd > 0)
	skiz->val[p] = 1;
      else
	skiz->val[p] = 0;
    }
  }

  DestroyAdjRel(&A);

  return(skiz);
}

Curve3D *CompSaliences(AnnImg *aimg, int maxcost)
{
  Image *cont=NULL;
  double *inter=NULL,*exter=NULL;  
  int p,n,i,Lmax;
  Curve3D *saliences=NULL;
  
  n       = aimg->img->ncols*aimg->img->nrows;
  cont = CreateImage(aimg->img->ncols,aimg->img->nrows);
  for(p=0;p<n; p++){
    if (aimg->pred->val[p]==p){
      cont->val[p]=aimg->label->val[p];
    }
  }
  
  Lmax    = MaximumValue(aimg->label);
  inter   = AllocDoubleArray(Lmax);
  exter   = AllocDoubleArray(Lmax);
  
  
  /* Compute influence areas */

  for (p=0; p < n; p++){
    if ((aimg->label->val[p] > 0)&&(aimg->cost->val[p] <= maxcost)) {
      if (aimg->img->val[p] != 0){
	inter[aimg->label->val[p]-1]++;
      } else {
	exter[aimg->label->val[p]-1]++;
      }
    }
  }
  
  /* Compute saliences */

  saliences  = CreateCurve3D(Lmax);
  
  for (p=0; p < n; p++){
    if (cont->val[p] > 0){
      i = cont->val[p]-1;
      saliences->X[i] = (double)(p%cont->ncols);
      saliences->Y[i] = (double)(p/cont->ncols);
      if (exter[i] > inter[i]){
	saliences->Z[i] = exter[i];
      }else{
	if (exter[i] < inter[i]){
	  saliences->Z[i] = -inter[i];
	}else{
	  saliences->Z[i] = 0.0;
	}
      }
    }
  }

  DestroyImage(&cont);
  free(inter);
  free(exter);

  return(saliences);
}


Curve3D *RemSaliencesByAngle(Curve3D *curve,int radius, int angle)
{
  Curve3D *scurve;
  double area;
  int i;

  scurve = CreateCurve3D(curve->n);
  for (i=0; i < curve->n; i++){ 
    scurve->X[i] = curve->X[i];
    scurve->Y[i] = curve->Y[i];
    scurve->Z[i] = curve->Z[i];
  }
  
  area = ((double)angle*PI*radius*radius/360.0);
  for (i=0; i < scurve->n; i++){ 
    if (fabs(scurve->Z[i]) <= area)
      scurve->Z[i] = 0.0;      
  }

  return(scurve);
}

Image *PaintSaliences(Image *bin, Curve3D *curve)
{
  int i,p;
  Image *points=NULL;

  points=ift_CopyImage(bin);
    
  for (i=0; i < curve->n; i++) {    
    p = (int)curve->X[i]+points->tbrow[(int)curve->Y[i]];
    if ((curve->Z[i] > 0.0)&&(points->val[p]!=0)){ /* Convex */
      PaintCircle(points,p,2.5,3);
    }  
    if ((curve->Z[i] < 0.0)&&(points->val[p]!=0)) { /* Concave */
      PaintCircle(points,p,2.5,2);
    }
  }
  return(points);
}

Image *PaintSkelSaliences(Image *skel, int angle){

  Curve3D *curve = NULL;
  Image *points = NULL;

  curve  = SkelSaliences(skel, 10, angle);
  points = PaintSaliences(skel, curve);

  DestroyCurve3D(&curve);
  return points;
}
  
void iftFastDilation(AnnImg *aimg, AdjRel *A)
{
  Image *Dx=NULL,*Dy=NULL;
  Queue *Q=NULL;
  Heap *H=NULL;
  int i,p,q,n,sz;
  Pixel u,v;
  int *sq=NULL,tmp=INT_MAX,dx,dy;
  bool cuisenaire;
  AdjRel *A8=Circular(1.5),*CA=NULL;
  char *color=NULL;

  if (aimg->seed == NULL)
    return;
  
  n  = MAX(aimg->img->ncols,aimg->img->nrows);
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;
  
  Dx = CreateImage(aimg->img->ncols,aimg->img->nrows);
  Dy = CreateImage(aimg->img->ncols,aimg->img->nrows);
  n  = aimg->img->ncols*aimg->img->nrows;
  color = AllocCharArray(n);
  sz = FrameSize(A);
  Q  = CreateQueue(2*sz*(sz+aimg->img->ncols+aimg->img->nrows),n);
  
  /* Compute IFT with 8-Adjacency */
  
  while (aimg->seed != NULL){
    p=RemoveSet(&(aimg->seed));
    InsertQueue(Q,aimg->cost->val[p]%Q->C.nbuckets,p);
    color[p]=GRAY;
  }
  
  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    color[p]=BLACK;
    u.x = p%aimg->img->ncols;
    u.y = p/aimg->img->ncols;
    cuisenaire=true;
    for (i=1; i < A8->n; i++){
      v.x = u.x + A8->dx[i];
      v.y = u.y + A8->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	q = v.x + aimg->img->tbrow[v.y];
	if (color[q] != BLACK){
	  dx  = Dx->val[p] + abs(v.x-u.x);
	  dy  = Dy->val[p] + abs(v.y-u.y);
	  tmp = sq[dx] + sq[dy];
	  if (tmp < aimg->cost->val[q]){
	    if (color[q] == WHITE){
	      InsertQueue(Q,tmp%Q->C.nbuckets,q);
	      color[q] = GRAY;
	      }else
		UpdateQueue(Q,q,aimg->cost->val[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	    aimg->cost->val[q]  = tmp;
	    aimg->pred->val[q]  = p;
	    aimg->label->val[q] = aimg->label->val[p];
	    Dx->val[q] = dx;
	    Dy->val[q] = dy;
	    cuisenaire = false;
	  }
	} 
      }    
    }
    if (cuisenaire)    
      InsertSet(&(aimg->seed),p); 
  }
  
  DestroyQueue(&Q);
  free(color);
  
  /* Compute IFT with Complementary Adjacency */
  
  if (A8->n < A->n) {
    
    CA = ComplAdj(A8,A);
    H  = CreateHeap(n,aimg->cost->val);

    while (aimg->seed != NULL){
      p=RemoveSet(&(aimg->seed));
      InsertHeap(H,p);
    }

    while(!HeapIsEmpty(H)) {
      RemoveHeap(H,&p);
      u.x = p%aimg->img->ncols;
      u.y = p/aimg->img->ncols;
      for (i=0; i < CA->n; i++){
	v.x = u.x + CA->dx[i];
	v.y = u.y + CA->dy[i];
	if (ValidPixel(aimg->img,v.x,v.y)){
	  q = v.x + aimg->img->tbrow[v.y];
	  if (color[q]!=BLACK){
	    dx  = Dx->val[p] + abs(v.x-u.x);
	    dy  = Dy->val[p] + abs(v.y-u.y);
	    tmp = sq[dx] + sq[dy];
	    if (tmp < aimg->cost->val[q]) 
	      {
		aimg->cost->val[q]  = tmp;
		aimg->pred->val[q]  = p;
		aimg->label->val[q] = aimg->label->val[p];
		Dx->val[q] = dx;
		Dy->val[q] = dy;
		if (color[q] == WHITE){
		  InsertHeap(H,q);
		}else
		  GoUpHeap(H,H->pos[q]);
	      }
	  }
	}    
      }
    }
    DestroyAdjRel(&CA);
    DestroyHeap(&H);
  }

  DestroyAdjRel(&A8);

  free(sq);
  DestroyImage(&Dx);
  DestroyImage(&Dy);
}

void iftDilation(AnnImg *aimg, AdjRel *A)
{
  Image *Dx=NULL,*Dy=NULL;
  Queue *Q=NULL;
  int i,p,q,n,sz;
  Pixel u,v;
  int *sq=NULL,tmp=INT_MAX,dx,dy;
  char *color=NULL;

  if (aimg->seed == NULL)
    return;

  n  = MAX(aimg->img->ncols,aimg->img->nrows);
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  Dx = CreateImage(aimg->img->ncols,aimg->img->nrows);
  Dy = CreateImage(aimg->img->ncols,aimg->img->nrows);
  n  = aimg->img->ncols*aimg->img->nrows;
  color = AllocCharArray(n);
  sz = FrameSize(A);
  Q  = CreateQueue(2*sz*(sz+aimg->img->ncols+aimg->img->nrows),n);
  
  while (aimg->seed != NULL){
    p=RemoveSet(&(aimg->seed));
    InsertQueue(Q,aimg->cost->val[p]%Q->C.nbuckets,p);
    color[p]=GRAY;
  }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    color[p]=BLACK;
    u.x = p%aimg->img->ncols;
    u.y = p/aimg->img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	q = v.x + aimg->img->tbrow[v.y];
	if (color[q] != BLACK){
	  dx  = Dx->val[p] + abs(v.x-u.x);
	  dy  = Dy->val[p] + abs(v.y-u.y);
	  tmp = sq[dx] + sq[dy];
	  if (tmp < aimg->cost->val[q])
	    {
	      if (color[q] == WHITE){
		InsertQueue(Q,tmp%Q->C.nbuckets,q);
		color[q]=GRAY;
	      }else
		UpdateQueue(Q,q,aimg->cost->val[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	      aimg->cost->val[q]  = tmp;
	      aimg->pred->val[q]  = p;
	      aimg->label->val[q] = aimg->label->val[p];
	      Dx->val[q] = dx;
	      Dy->val[q] = dy;
	    }
	}
      }    
    }
  }
  free(color);
  free(sq);
  DestroyQueue(&Q);
  DestroyImage(&Dx);
  DestroyImage(&Dy);

}

void iftDilationHeap(AnnImg *aimg, AdjRel *A)
{
  Image *Dx=NULL,*Dy=NULL;
  Heap *H=NULL;
  int i,p,q,n;
  Pixel u,v;
  int *sq=NULL,tmp=INT_MAX,dx,dy;

  if (aimg->seed == NULL)
    return;

  n  = MAX(aimg->img->ncols,aimg->img->nrows);
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  Dx = CreateImage(aimg->img->ncols,aimg->img->nrows);
  Dy = CreateImage(aimg->img->ncols,aimg->img->nrows);
  n  = aimg->img->ncols*aimg->img->nrows;
  H  = CreateHeap(n,aimg->cost->val);
  
  while (aimg->seed != NULL){
    p=RemoveSet(&(aimg->seed));
    InsertHeap(H,p);
  }
  
  while(!HeapIsEmpty(H)) {
    RemoveHeap(H,&p);
    u.x = p%aimg->img->ncols;
    u.y = p/aimg->img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	q = v.x + aimg->img->tbrow[v.y];
	if (H->color[q] != BLACK){
	  dx  = Dx->val[p] + abs(v.x-u.x);
	  dy  = Dy->val[p] + abs(v.y-u.y);
	  tmp = sq[dx] + sq[dy];
	  if (tmp < aimg->cost->val[q]){
	      aimg->cost->val[q]  = tmp;
	      aimg->pred->val[q]  = p;
	      aimg->label->val[q] = aimg->label->val[p];
	      Dx->val[q] = dx;
	      Dy->val[q] = dy;

	      if (H->color[q] == WHITE)
		InsertHeap(H,q);
	      else
		GoUpHeap(H,H->pos[q]);
	    }
	}
      }    
    }
  }

  free(sq);
  DestroyHeap(&H);
  DestroyImage(&Dx);
  DestroyImage(&Dy);

}

int iftGeoDilation(AnnImg *aimg) /* Chamfer 17x24 */
{
  Queue *Q=NULL;
  Pixel u,v;
  int i,dst=-1,p,q,tmp,c;
  AdjRel *A=NULL;

  A = Circular(1.5);
  Q = CreateQueue(25,aimg->img->ncols*aimg->img->nrows);

  while (aimg->seed != NULL){
    p=RemoveSet(&(aimg->seed));
    if (aimg->label->val[p]!=0)
      InsertQueue(Q,aimg->cost->val[p]%Q->C.nbuckets,p);
    else
      aimg->cost->val[p] = INT_MAX;
  }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (aimg->label->val[p]==2){
      dst = p;
      break;
    }
    u.x = p%aimg->img->ncols;
    u.y = p/aimg->img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	q = v.x + aimg->img->tbrow[v.y];
	if ((aimg->cost->val[p] < aimg->cost->val[q])&&(aimg->img->val[q]!=0)){
	  if ((i%2)==0)
	    c = 24;
	  else
	    c = 17;	  
	  tmp = aimg->cost->val[p] + c;
	  if (tmp < aimg->cost->val[q]){
	    if (aimg->cost->val[q] == INT_MAX)
	      InsertQueue(Q,tmp%Q->C.nbuckets,q);
	    else
	      UpdateQueue(Q,q,aimg->cost->val[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	    aimg->cost->val[q]  = tmp;
	    aimg->pred->val[q]  = p;
	    if (aimg->label->val[q] != 2) 
	      aimg->label->val[q] = aimg->label->val[p];
	  }
	}
      }    
    }
  }

  while(!EmptyQueue(Q)){
    p = RemoveQueue(Q);
    InsertSet(&(aimg->seed),p);
  }

  DestroyQueue(&Q);
  DestroyAdjRel(&A);

  return(dst);
}

/*
Image *iftDistTrans(AnnImg *aimg, AdjRel *A, char side)
{
  int p,n;

  n = aimg->img->ncols*aimg->img->nrows;

  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++)
      if (aimg->img->val[p] == 0)
	aimg->cost->val[p] = 0;
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++)
      if ((aimg->img->val[p] != 0)&&(aimg->cost->val[p]==INT_MAX))
	aimg->cost->val[p] = 0;
    break;
  }

  iftDilation(aimg, A);
  return(ift_CopyImage(aimg->cost));
}
*/

Image *DistTrans(Image *bin, AdjRel *A, char side)
{
  Image *Dx=NULL,*Dy=NULL,*fbin,*fcont,*fcost,*cost;
  Queue *Q=NULL;
  int i,p,q,n,sz;
  Pixel u,v;
  int *sq=NULL,tmp=INT_MAX,dx,dy;
  AdjPxl *N;

  n  = MAX(bin->ncols,bin->nrows);
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  sz = FrameSize(A);  
  fbin = AddFrame(bin,sz,0);
  fcont = ObjectBorder(fbin);
  fcost = AddFrame(bin,sz,INT_MIN);
  Dx = CreateImage(fcost->ncols,fcost->nrows);
  Dy = CreateImage(fcost->ncols,fcost->nrows);  
  N  = AdjPixels(fcost,A);
  n  = fcost->ncols*fcost->nrows;
  Q = CreateQueue(2*sz*(sz+bin->ncols+bin->nrows),n);

  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->val[p] != 0){
	if (fcont->val[p]>0){
	  fcost->val[p]=0;    
	  InsertQueue(Q,fcost->val[p]%Q->C.nbuckets,p);
	} else
	  fcost->val[p] = INT_MAX;	  
      }else{
	if (fcost->val[p]!=INT_MIN)
	  fcost->val[p] = 0;
      }
    }
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->val[p] == 0){
	if (fcost->val[p]!=INT_MIN)
	  fcost->val[p] = INT_MAX;	  
      }else{
	if (fcont->val[p]>0){
	  fcost->val[p]=0;    
	  InsertQueue(Q,fcost->val[p]%Q->C.nbuckets,p);
	}else
	  fcost->val[p] = 0;
      }
    }
    break;
  case BOTH:
  default:    
    for(p = 0; p < n; p++){
      if (fcont->val[p] > 0){
	fcost->val[p]=0;    
	InsertQueue(Q,fcost->val[p]%Q->C.nbuckets,p);
      }else{ 
	if (fcost->val[p]!=INT_MIN)
	  fcost->val[p]=INT_MAX;    
      }
    }
  }
  
  DestroyImage(&fcont);
  DestroyImage(&fbin);
  
  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] < fcost->val[q]){
	u.x = p%fcost->ncols;
	u.y = p/fcost->ncols;
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	dx  = Dx->val[p] + abs(v.x-u.x);
	dy  = Dy->val[p] + abs(v.y-u.y);
	tmp = sq[dx] + sq[dy];
	if (tmp < fcost->val[q]){
	  if (fcost->val[q] == INT_MAX)
	    InsertQueue(Q,tmp%Q->C.nbuckets,q);
	  else
	    UpdateQueue(Q,q,fcost->val[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	  fcost->val[q]  = tmp;
	  Dx->val[q] = dx;
	  Dy->val[q] = dy;
	}
      }
    }
  }

  DestroyQueue(&Q);
  DestroyAdjPxl(&N);
  cost = RemFrame(fcost,sz);

  free(sq);
  DestroyImage(&Dx);
  DestroyImage(&Dy);
  DestroyImage(&fcost);
  
  return(cost);
}

Image *SignedDistTrans(Image *bin, AdjRel *A, char side)
{
  Image *Dx=NULL,*Dy=NULL,*fbin,*fcont,*fcost,*cost;
  Queue *Q=NULL;
  int i,p,q,n,sz;
  Pixel u,v;
  int *sq=NULL,tmp=INT_MAX,dx,dy;
  AdjPxl *N;

  n  = MAX(bin->ncols,bin->nrows);
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  sz = FrameSize(A);  
  fbin = AddFrame(bin,sz,0);
  fcont = ObjectBorder(fbin);
  fcost = AddFrame(bin,sz,INT_MIN);
  Dx = CreateImage(fcost->ncols,fcost->nrows);
  Dy = CreateImage(fcost->ncols,fcost->nrows);  
  N  = AdjPixels(fcost,A);
  n  = fcost->ncols*fcost->nrows;
  Q = CreateQueue(2*sz*(sz+bin->ncols+bin->nrows),n);

  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->val[p] != 0){
	if (fcont->val[p]>0){
	  fcost->val[p]=0;    
	  InsertQueue(Q,fcost->val[p]%Q->C.nbuckets,p);
	} else
	  fcost->val[p] = INT_MAX;	  
      }else{
	if (fcost->val[p]!=INT_MIN)
	  fcost->val[p] = 0;
      }
    }
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->val[p] == 0){
	if (fcost->val[p]!=INT_MIN)
	  fcost->val[p] = INT_MAX;	  
      }else{
	if (fcont->val[p]>0){
	  fcost->val[p]=0;    
	  InsertQueue(Q,fcost->val[p]%Q->C.nbuckets,p);
	}else
	  fcost->val[p] = 0;
      }
    }
    break;
  case BOTH:
  default:    
    for(p = 0; p < n; p++){
      if (fcont->val[p] > 0){
	fcost->val[p]=0;    
	InsertQueue(Q,fcost->val[p]%Q->C.nbuckets,p);
      }else{ 
	if (fcost->val[p]!=INT_MIN)
	  fcost->val[p]=INT_MAX;    
      }
    }
  }
  
  DestroyImage(&fcont);
  DestroyImage(&fbin);
  
  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] < fcost->val[q]){
	u.x = p%fcost->ncols;
	u.y = p/fcost->ncols;
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	dx  = Dx->val[p] + abs(v.x-u.x);
	dy  = Dy->val[p] + abs(v.y-u.y);
	tmp = sq[dx] + sq[dy];
	if (tmp < fcost->val[q]){
	  if (fcost->val[q] == INT_MAX)
	    InsertQueue(Q,tmp%Q->C.nbuckets,q);
	  else
	    UpdateQueue(Q,q,fcost->val[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	  fcost->val[q]  = tmp;
	  Dx->val[q] = dx;
	  Dy->val[q] = dy;
	}
      }
    }
  }

  DestroyQueue(&Q);
  DestroyAdjPxl(&N);
  cost = RemFrame(fcost,sz);
  // sign image
  n  = cost->ncols*cost->nrows;  

  if (side != INTERIOR)
    for (i=0; i<n; i++) {
      if (bin->val[i] == 0) {
	cost->val[i] = -cost->val[i];
      }
    }
  free(sq);
  DestroyImage(&Dx);
  DestroyImage(&Dy);
  DestroyImage(&fcost);
  
  return(cost);
}

int iftGeoDist(AnnImg *aimg)
{
  int dist=INT_MAX,p;
  
  p    = iftGeoDilation(aimg);
  if (p != -1)
    dist = aimg->cost->val[p];
  return(dist);
}

int *iftGeoPath(AnnImg *aimg)
{
  int p, *path = NULL;

  p    = iftGeoDilation(aimg);
  if (p != -1)
    path = Path(aimg->pred,p);
  return(path);
}

Polynom *MSFractal(Image *bin, 
		   int maxdist, 
		   int degree, 
		   double lower, 
		   double higher,
		   int reg,
		   double from,
		   double to)
{
  Curve *hist=NULL,*haux=NULL,*ahist=NULL, *aux_ahist=NULL,*loglog=NULL;
  AnnImg *aimg=NULL;
  AdjRel *A=NULL;
  Image *mbb=NULL,*nbin=NULL;
  Polynom *P=NULL,*D=NULL;
  int n,i,j,maxcost=maxdist*maxdist;
    
  mbb  = MBB(bin);
  nbin = AddFrame(mbb,maxdist,0);
  DestroyImage(&mbb);

  /* Compute Euclidean IFT */

  A = Circular(1.5);
  aimg = Annotate(nbin,NULL,nbin);
  iftDilation(aimg, A);
  DestroyAdjRel(&A);

  /* Compute MS Fractal */

  hist = Histogram(aimg->cost);

  /* Compute non-zero points */

  n = 0;
  for (i=1; i < maxcost; i++)
    if (hist->Y[i] != 0)
      n++;

  haux = CreateCurve(n);
  j=0;
  for (i=1; i < maxcost; i++)
    if (hist->Y[i] != 0){
      haux->X[j] = log(sqrt((double)i));
      haux->Y[j] = hist->Y[i];
      j++;
    }
  
  /* Accumulate values */
  ahist = CreateCurve(n);
  ahist->X[0] = haux->X[0];
  ahist->Y[0] = haux->Y[0];
  for (i=1; i < n; i++) {
    ahist->X[i] = haux->X[i];
    ahist->Y[i] = ahist->Y[i-1] + haux->Y[i];
  }

  /* Compute log(Y) */
  for (i=0; i < n; i++)
    ahist->Y[i] = log((double)ahist->Y[i]);
  
  j=0;
 
  for (i=0; i < n; i++)
    if ((ahist->X[i]>from)&&((ahist->X[i]<to)))
      j++;
  
  aux_ahist = CreateCurve(j);
  
  j=0;
  for (i=0; i < n; i++)
    if ((ahist->X[i]>from)&&((ahist->X[i]<to))){
      aux_ahist->X[j] = ahist->X[i];
      aux_ahist->Y[j] = ahist->Y[i];
      j++;
    }
  
  
  /* Compute Regression */
  P = Regression(/*ahist*/aux_ahist,degree);
  
  /* Print loglog curve */
  if (reg){
    loglog = SamplePolynom(P,lower, higher, 100);
    WriteCurve(ahist, "experimental");
    WriteCurve(loglog, "regression");
  }
  
  /* Compute Fractal Curve */
  D = DerivPolynom(P);  

  DestroyCurve(&hist);
  DestroyCurve(&haux);
  DestroyCurve(&ahist);
  DestroyCurve(&aux_ahist);
  DestroyCurve(&loglog);
  DestroyPolynom(&P);
  DeAnnotate(&aimg);
  DestroyImage(&nbin);
  return(D);
}

/*
Curve *MSFractal(Image *shape, int maxdist)
{
  Curve *fractal=NULL,*hist=NULL,*haux=NULL,*ahist=NULL;
  AnnImg *aimg=NULL;
  AdjRel *A=NULL;
  Image *mbb=NULL,*nbin=NULL;
  int n,i,j,maxcost=maxdist*maxdist;
  Polynom *P=NULL,*D=NULL;
  double lower=MAX(0.1*log(maxdist),0.5), higher=MAX(0.9*log(maxdist),lower+5.0), inc=(higher-lower)/100.0;

  mbb  = MBB(shape);
  nbin = AddFrame(mbb,maxdist,0);
  DestroyImage(&mbb);
*/
  /* Compute Euclidean IFT */
/*
  A = Circular(1.5);
  aimg = Annotate(nbin,NULL,nbin);
  n = aimg->img->ncols*aimg->img->nrows;
  iftDilation(aimg, A);
  DestroyAdjRel(&A);
*/
  /* Compute MS Fractal */
/*
  hist = Histogram(aimg->cost);
*/
  /* Compute non-zero points */
/*
  n = 0;
  for (i=1; i < maxcost; i++)
    if (hist->Y[i] != 0)
      n++;

  haux = CreateCurve(n);
  j=0;
  for (i=1; i < maxcost; i++)
    if (hist->Y[i] != 0){
      haux->X[j] = log(sqrt((double)i));
      haux->Y[j] = hist->Y[i];
      j++;
    }
*/
  /* Accumulate values */
/*
  ahist = CreateCurve(n);
  ahist->X[0] = haux->X[0];
  ahist->Y[0] = haux->Y[0];
  for (i=1; i < n; i++) {
    ahist->X[i] = haux->X[i];
    ahist->Y[i] = ahist->Y[i-1] + haux->Y[i];
  }
*/
  /* Compute log(Y) */
/*
  for (i=0; i < n; i++)
  ahist->Y[i] = log((double)ahist->Y[i]);
*/  
/* Compute Regression */
/*
  P = Regression(ahist,10);
*/
/* Compute Fractal Curve */
/*
  D = DerivPolynom(P);

  fractal = SamplePolynom(D,lower,higher,inc);
  
  for (i=0; i < fractal->n; i++) {
    fractal->Y[i] = 2.0 - fractal->Y[i];
  }

  DestroyCurve(&hist);
  DestroyCurve(&haux);
  DestroyCurve(&ahist);
  DestroyPolynom(&P);
  DestroyPolynom(&D);
  DeAnnotate(&aimg);
  DestroyImage(&nbin);
  
  return(fractal);
}
*/
Image *MSSkel(Image *bin, char side)
{
  Image *msskel,*cont;
  AdjRel *A;
  AnnImg *aimg;
  int p,n;

  /* Compute Euclidean IFT */

  cont   = LabelContPixel(bin);
  aimg   = Annotate(bin,NULL,cont); 
  A      = Circular(1.5);
  n      = aimg->img->ncols*aimg->img->nrows;
  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++)
      if (aimg->img->val[p] == 0){
	aimg->cost->val[p] = 0;
      }
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++)
      if (aimg->img->val[p] != 0){
	aimg->cost->val[p] = 0;
      }
    break;
  case BOTH:
  default:    
    ;
  }
  iftDilation(aimg,A);
  DestroyAdjRel(&A);
  DestroyImage(&cont);
  
  /* Compute MS Skeletons */

  msskel = CompMSSkel(aimg);
  DeAnnotate(&aimg);

  return(msskel);
}

Image *SKIZ(Image *bin, char side)
{
  Image *skiz,*cont;
  AdjRel *A;
  AnnImg *aimg;
  int p,n;

  /* Compute Euclidean IFT */

  cont   = LabelContour(bin);
  aimg   = Annotate(bin,NULL,cont); 
  A      = Circular(1.5);
  n      = aimg->img->ncols*aimg->img->nrows;
  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++)
      if (aimg->img->val[p] == 0){
	aimg->cost->val[p] = 0;
      }
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++)
      if (aimg->img->val[p] != 0){
	aimg->cost->val[p] = 0;
      }
    break;
  case BOTH:
  default:    
    ;
  }
  iftDilation(aimg,A);
  DestroyAdjRel(&A);
  DestroyImage(&cont);
  
  /* Compute SKIZ */
  
  skiz = CompSKIZ(aimg);
  DeAnnotate(&aimg);
  
  return(skiz);
}

Image *LabelSkel(Image *skel, Curve3D *curve, char option)
{
  
  Image *mainbody = NULL;
  Image *label = NULL;
  Image *dist = NULL;
  Image *parent = NULL;
  Image *length = NULL;
  Image *output = NULL;
  char  *color = NULL;
  AdjRel *A;
  Queue *Q;
  Pixel u, v;
  int p, q, r, i, j, n, maxlabel, maxdist;
  
  n = skel->ncols*skel->nrows;
  color      = AllocCharArray(n);
  mainbody   = CreateImage(skel->ncols,skel->nrows);
  label      = CreateImage(skel->ncols,skel->nrows);
  dist       = CreateImage(skel->ncols,skel->nrows);
  parent     = CreateImage(skel->ncols,skel->nrows);
  length     = CreateImage(skel->ncols,skel->nrows);
  A          = Circular(1.5);
  for (p=0; p<n; p++){
    if (skel->val[p]!=0){
      j = 0;
      u.x = p%skel->ncols;
      u.y = p/skel->ncols;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(skel,v.x,v.y)){
	  q = v.x + skel->tbrow[v.y];
	  if (skel->val[q]!=0)
	    j++;
	}
      }
      if (j>3)
	mainbody->val[p] = 1;
    }
  }
  DestroyAdjRel(&A); 
  
  for (i=0; i<curve->n; i++){
    if (curve->Z[i]>0.0){
      p = (int)curve->X[i]+skel->tbrow[(int)curve->Y[i]];
      label->val[p]  = (int)curve->Z[i];
      dist->val[p]   = 1;
      parent->val[p] = p;
    }
  }
  
  maxlabel = MaximumValue(label);
  
  Q        = CreateQueue(maxlabel+2,n);
  A        = Circular(1.5); 
  
  for (p=0; p<n; p++)
    if (label->val[p] !=0){
      color[p]=GRAY;
      InsertQueue(Q,label->val[p],p);
    }
  
  while(!EmptyQueue(Q)) {
    p = RemoveQueue(Q);
    color[p]=BLACK;
    n = 0;
    u.x = p%skel->ncols;
    u.y = p/skel->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(skel,v.x,v.y)){
	q = v.x + skel->tbrow[v.y];
	if (skel->val[q]!=0){
	  if (color[q] != BLACK){
	    if (mainbody->val[q]==0){
	      if (color[q] == WHITE){
		n++;
		color[q]=GRAY;
		InsertQueue(Q,label->val[p],q);
		label->val[q] = label->val[p];
		dist->val[q] = dist->val[p] + 1;
		parent->val[q] = p;
	      }
	    }
	    else{
	      r = p;
	      while (r!=parent->val[r]){
		length->val[r] = dist->val[p];
		r = parent->val[r];
	      }
	      length->val[r] = dist->val[p];
		
	      if (color[q] == WHITE){
		n++;
		color[q] = GRAY;
		InsertQueue(Q, maxlabel+1,q);
		label->val[q] = maxlabel+1;
		dist->val[q] = dist->val[p] + 1;
	      }
	    }
	  }
	}
      }
    }
    if (n==0){
      r = p;
      while (r!=parent->val[r]){
	length->val[r] = dist->val[p];
	r = parent->val[r];
      }
      length->val[r] = dist->val[p];
    }
  }
  
  maxdist = MaximumValue(dist);
  n = skel->ncols*skel->nrows;
  for (p=0; p<n; p++)
    if (label->val[p]==(maxlabel+1)){
      dist->val[p] = maxdist + 1;
      length->val[p] = maxdist + 1;
    }
      
  switch (option) {
  case 0:
    output = ift_CopyImage(label);
    break;
  case 1:
    output = ift_CopyImage(dist);
    break;
  case 2:
    output = ift_CopyImage(length);
    break;
  }

  DestroyAdjRel(&A); 
  DestroyQueue(&Q);
  DestroyImage(&mainbody);
  DestroyImage(&label);
  DestroyImage(&dist);
  DestroyImage(&parent);
  DestroyImage(&length);
  free(color);
  return(output);
}
 
Curve3D *ContSaliences(Image *bin,int maxdist,int threshold,int angle)
{
  int x, y, i;
  Pixel right, left;
  Curve3D *cont_salie = NULL;
  Curve3D *salie_int = NULL;   
  Curve3D *salie_ext = NULL; 
  Curve3D *saliences = NULL;
    
  salie_int = SkelCont(bin, maxdist, threshold, angle, INTERIOR);
  salie_ext = SkelCont(bin, maxdist, (int)2*threshold, angle, EXTERIOR);
    
  left.x  = bin->ncols-1;
  left.y  = bin->nrows-1;
  right.x = 0;
  right.y = 0;
  for (y=0; y < bin->nrows; y++)
    for (x=0; x < bin->ncols; x++){
      if (bin->val[x+bin->tbrow[y]] > 0){
	if (x < left.x)
	  left.x = x;
	if (y < left.y)
	  left.y = y;
	if (x > right.x)
	  right.x = x;
	if (y > right.y)
	  right.y = y;	
      }
    }

    
  for (i=0; i<salie_ext->n; i++){
    if ((salie_ext->X[i]<left.x)||
	(salie_ext->X[i]>right.x)||
	(salie_ext->Y[i]<left.y)||
	(salie_ext->Y[i]>right.y))
      salie_ext->Z[i] = 0.0;
  }
  
  cont_salie = Saliences(bin, maxdist);
  saliences  = CreateCurve3D(cont_salie->n);
  for (i=0; i<salie_int->n; i++){
    if (salie_int->Z[i]>0.0){
      saliences->X[(int)salie_int->Z[i]-1] = cont_salie->X[(int)salie_int->Z[i]-1];
      saliences->Y[(int)salie_int->Z[i]-1] = cont_salie->Y[(int)salie_int->Z[i]-1];
      saliences->Z[(int)salie_int->Z[i]-1] = cont_salie->Z[(int)salie_int->Z[i]-1];      
    }
  }
  for (i=0; i<salie_ext->n; i++){
    if (salie_ext->Z[i]>0.0){
      saliences->X[(int)salie_ext->Z[i]-1] = cont_salie->X[(int)salie_ext->Z[i]-1];
      saliences->Y[(int)salie_ext->Z[i]-1] = cont_salie->Y[(int)salie_ext->Z[i]-1];
      saliences->Z[(int)salie_ext->Z[i]-1] = cont_salie->Z[(int)salie_ext->Z[i]-1];
    }
  }

  DestroyCurve3D(&cont_salie);
  DestroyCurve3D(&salie_int);   
  DestroyCurve3D(&salie_ext);
  return(saliences);
}

Curve3D *Saliences(Image *bin, int maxdist) 
{  
  Image *cont=NULL;
  AdjRel *A=NULL;
  AnnImg *aimg=NULL;
  Curve3D *saliences=NULL;
  
  /* Compute Euclidean IFT */
  
  cont    = LabelContPixel(bin);
  aimg    = Annotate(bin,NULL,cont); 
  A       = Circular(1.5);
  iftDilation(aimg,A);
  saliences = CompSaliences(aimg, maxdist*maxdist);
  
  DestroyImage(&cont);
  DestroyAdjRel(&A);
  DeAnnotate(&aimg);
  
  return(saliences);
}

Curve3D *SkelSaliences(Image *skel, int maxdist, int angle) 
{  
  Image *cont=NULL;
  AdjRel *A=NULL;
  AnnImg *aimg=NULL;
  Curve3D *saliences=NULL;
  Curve3D *auxsalie=NULL;

  /* Compute Euclidean IFT */
  A         = Circular(0.0);
  cont      = LabelBinComp(skel, A);
  aimg      = Annotate(skel,NULL,cont);
  DestroyAdjRel(&A); 

  A         = Circular(1.5);
  iftDilation(aimg,A);
  auxsalie  = CompSaliences(aimg, maxdist*maxdist);
  saliences = RemSaliencesByAngle(auxsalie,maxdist,angle);
  DestroyImage(&cont);
  DestroyAdjRel(&A);
  DeAnnotate(&aimg);
  DestroyCurve3D(&auxsalie);
  return(saliences);
}

Curve3D *SkelCont(Image *bin, int maxdist, int threshold, int angle, char side) {  
  Image   *contour=NULL;
  Image   *msskel=NULL;
  Image   *skel=NULL;
  Image   *bin_skel=NULL;
  AdjRel  *A=NULL;
  AnnImg  *aimg=NULL;
  Curve3D *contour_salie = NULL;
  Curve3D *skelsaliences = NULL;
  Curve3D *saliences = NULL;
  Pixel left, right;
  int i, j, p, q, n, label, imax, maxcont, max, min, imin, x, y, ne, ni, delta = 3; 
  double sum;
  
  contour   = LabelContPixel(bin);
  aimg      = Annotate(bin,NULL,contour); 
  A         = Circular(1.5);
  iftFastDilation(aimg,A);
  
  msskel   = MSSkel(bin, side);
  skel     = Skeleton(msskel, threshold);
  bin_skel = Skeleton(msskel, threshold);
  n        = bin->ncols*bin->nrows;
  contour_salie = Saliences(bin, maxdist);

  maxcont = MaximumValue(contour);

  for (p=0; p<n; p++){
    if (skel->val[p]!=0){
      q = Seed(aimg->pred, p);
      label = (aimg->label->val[q] + msskel->val[p]/2 + maxcont)%maxcont;
      skel->val[p] = label;
      if (side == INTERIOR){
	if (contour_salie->Z[label-1]<0.0){
	  max = INT_MIN;
	  imax = 0;
	  for (j=-5; j<5; j++){
	    if (contour_salie->Z[(label-1+j+contour_salie->n)%contour_salie->n] > max){
	      imax = (label-1+j+contour_salie->n)%contour_salie->n;
	      max = contour_salie->Z[imax];
	    }
	  }
	  skel->val[p] = imax + 1;
	}
	else {
	  skel->val[p] = MAX(label,1);
	}
      }
      else{ 
	if (side == EXTERIOR){
	  if (contour_salie->Z[label-1]>0.0){
	    min = INT_MAX;
	    imin = 0;
	    for (j=-5; j<5; j++){
	      if (contour_salie->Z[(label-1+j+contour_salie->n)%contour_salie->n] < min){
		imin = (label-1+j+contour_salie->n)%contour_salie->n;
		min = contour_salie->Z[imin];
	      }
	    }
	    skel->val[p] = imin + 1;
	  }
	  else {
	    skel->val[p] = MAX(label, 1);
	  }
	}
      }
    }
  }
  
  skelsaliences = SkelSaliences(bin_skel, maxdist, angle); 
  
  if (side==EXTERIOR){
    left.x  = bin->ncols-1;
    left.y  = bin->nrows-1;
    right.x = 0;
    right.y = 0;
    for (y=0; y < bin->nrows; y++)
      for (x=0; x < bin->ncols; x++){
	if (bin->val[x+bin->tbrow[y]] > 0){
	  if (x < left.x)
	    left.x = x;
	  if (y < left.y)
	    left.y = y;
	  if (x > right.x)
	    right.x = x;
	  if (y > right.y)
	    right.y = y;	
	}
      }
    
    for (i=0; i<skelsaliences->n; i++){
      if ((skelsaliences->X[i]<left.x)||
	  (skelsaliences->X[i]>right.x)||
	  (skelsaliences->Y[i]<left.y)||
	  (skelsaliences->Y[i]>right.y))
	skelsaliences->Z[i] = 0.0;
    }
  }
  
  SortCurve3D(skelsaliences, 0, (skelsaliences->n - 1), DECREASING);
  i=0;
  while (skelsaliences->Z[i]!=0.0)
    i++;
  saliences = CreateCurve3D(i);  
  for (i=0; i< saliences->n; i++){
    if (skelsaliences->Z[i]!=0.0){
      p = (int)skelsaliences->X[i]+bin->tbrow[(int)skelsaliences->Y[i]];
      saliences->X[i] = contour_salie->X[skel->val[p]-1];
      saliences->Y[i] = contour_salie->Y[skel->val[p]-1];
      if (side==INTERIOR){
	sum = 0.0;
	for (j=-delta; j<=delta; j++){
	  q = ((skel->val[p]-1) + j + maxcont) % maxcont;
	  if (contour_salie->Z[q]>0.0)
	    sum += contour_salie->Z[q];
	}
	saliences->Z[i] = sum;
      }      
      else{
	sum = 0.0;
	for (j=-delta; j<=delta; j++){
	  q = ((skel->val[p]-1) + j + maxcont) % maxcont;
	  if (contour_salie->Z[q]<0.0)
	    sum += contour_salie->Z[q];
	}
	saliences->Z[i] = sum;
      }
    }  
  }

  ne = 0;
  ni = 0;
  for (i=0; i<saliences->n; i++){
    if (saliences->Z[i]>0.0) 
      ni += saliences->Z[i];
    else
      if (saliences->Z[i]<0.0) 
	ne += fabs(saliences->Z[i]);
  }

  for (i=0; i<saliences->n; i++){
    if (saliences->Z[i]>0.0) 
      saliences->Z[i]/=ni;
    else
      if (saliences->Z[i]<0.0) 
	saliences->Z[i]/=ne;
  }

  
  DestroyImage(&contour);
  DestroyImage(&msskel);
  DestroyImage(&skel);
  DestroyImage(&bin_skel);
  DestroyCurve3D(&contour_salie);
  DestroyCurve3D(&skelsaliences);
  DestroyAdjRel(&A);
  DeAnnotate(&aimg);
  return(saliences);

}



/**
 * Funcao auxiliar que extrai o contorno do objeto na imagem binária.
 * Assume que a imagem contém somente um objeto.
 * Baseada na função LabelContour.
 **/
static int *ExtractContourBin(Image *bin, int *contour_size) {

  Image *bndr=NULL, *color=NULL, *pred=NULL;
  int p=0,q,r,i,j,n,left=0,right=0,*LIFO,last,l;
  int *contour;
  AdjRel *A,*L,*R;
  Pixel u,v,w;
  
  A     = Circular(1.0);
  n     = bin->ncols * bin->nrows;
  bndr  = CreateImage(bin->ncols, bin->nrows);

  *contour_size = 0;

  /* marca na imagem bndr somente o contorno do objeto com 4-vizinhos */
  for (p=0; p < n; p++) {
    if (bin->val[p] != 0) {
      u.x = p % bin->ncols;
      u.y = p / bin->ncols;
      for (i = 1; i < A->n; i++) {
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(bin,v.x,v.y)) {
	  q = v.x + bin->tbrow[v.y];
	  if (bin->val[q] == 0) {
	    bndr->val[p] = 1;
	    break;
	  }
	} else {
	    bndr->val[p] = 1;
	    break;
	}
      }
    }
  }  
  DestroyAdjRel(&A);

  A      = Circular(1.5);
  L      = LeftSide(A);
  R      = RightSide(A);
  color  = CreateImage(bin->ncols, bin->nrows);
  pred   = CreateImage(bin->ncols, bin->nrows);
  LIFO   = AllocIntArray(n);
  last   = NIL;

  /* avança até encontrar um ponto na máscara */
  for (j = 0; j < n && (bndr->val[j] == 0
			|| !ValidContPoint(bin, L, R, j)); j++);

  if (j == n)
    return NULL; /* imagem vazia */

  last++;
  LIFO[last]    = j;
  color->val[j] = GRAY;
  pred->val[j]  = j;
  while (last != NIL) {
    p = LIFO[last]; last--;	
    color->val[p]=BLACK;
    u.x = p%bndr->ncols;
    u.y = p/bndr->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(bndr,v.x,v.y)){
	q = v.x + bndr->tbrow[v.y];
	if ((q==j)&&(pred->val[p]!=j)){
	  last = NIL;
	  break;
	}
	
	w.x = u.x + L->dx[i]; 
	w.y = u.y + L->dy[i];
	if (ValidPixel(bndr,w.x,w.y))
	  left = w.x + bndr->tbrow[w.y];
	else
	  left = -1;
	w.x = u.x + R->dx[i]; 
	w.y = u.y + R->dy[i];
	if (ValidPixel(bndr,w.x,w.y))
	  right = w.x + bndr->tbrow[w.y];
	else
	  right = -1;
	
	if ((bndr->val[q]==1)&&
	    (color->val[q] != BLACK)&&
	    (((left!=-1)&&(right!=-1)&&(bin->val[left]!=bin->val[right]))||
	     ((left==-1)&&(right!=-1)&&(bin->val[right]==1)) ||
	     ((right==-1)&&(left!=-1)&&(bin->val[left]==1)))) {
	  pred->val[q] = p;
	  (*contour_size)++;
	  if (color->val[q] == WHITE){
	    last++;
	    LIFO[last] = q;
	    color->val[q]=GRAY;
	  }
	} 
      }
    }	
  }

  contour = (int *) malloc(sizeof(int) * (*contour_size));

  r = p;
  for (l = 0; pred->val[p] != p; ++l, p = pred->val[p])
    contour[l] = p;
  if (r != p)
    contour[l] = p;

  DestroyAdjRel(&A);
  DestroyAdjRel(&L);
  DestroyAdjRel(&R);
  DestroyImage(&bndr);
  DestroyImage(&color);
  DestroyImage(&pred);
  free(LIFO);

  return contour;
}

/**
 * Calcula a distancia geodésica entre dois pixels na imagem binária.
 * Modificação de iftGeoDilation para receber diretamente os
 * parametros e nao gerar imagens de label e predecessores.
 * Chamfer 17x24
 **/
static int GeoDilationBin(Image *img, int start, int end) {

  Queue *Q=NULL;
  Pixel u,v;
  int i,p,q,tmp,c;
  AdjRel *A=NULL;
  Image *cost;
  int npixels;

  A = Circular(1.5);
  npixels = img->ncols * img->nrows;
  Q = CreateQueue(25, npixels);
  cost = CreateImage(img->ncols, img->nrows);

  for (i = 0; i < npixels; ++i)
    cost->val[i] = INT_MAX;

  cost->val[start] = 0;
  InsertQueue(Q, 0, start);  

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);

    if (p == end)
      break;

    u.x = p % img->ncols;
    u.y = p / img->ncols;
    for (i = 1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img, v.x, v.y)) {
	q = v.x + img->tbrow[v.y];
	if ((cost->val[p] < cost->val[q]) && (img->val[q] != 0)) {
	  if ((i % 2) == 0)
	    c = 24;
	  else
	    c = 17;
	  tmp = cost->val[p] + c;
	  if (tmp < cost->val[q]) {
	    if (cost->val[q] == INT_MAX)
	      InsertQueue(Q, tmp % Q->C.nbuckets, q);
	    else
	      UpdateQueue(Q, q, cost->val[q] % Q->C.nbuckets,
			  tmp % Q->C.nbuckets);
	    cost->val[q] = tmp;
	  }
	}
      }
    }
  }
  
  p = cost->val[end];

  DestroyQueue(&Q);
  DestroyAdjRel(&A);
  DestroyImage(&cost);
  return p;
}

/**
 * Calcula a maior e menor distância geodésica entre pontos
 * opostos do contorno.
 **/

void GeoDistMaxMinBin(Image *bin, int *max, int *min) {

  register int i, per, dist, mindist, maxdist;
  int *contour, contour_size;

  contour = ExtractContourBin(bin, &contour_size);

  maxdist = INT_MIN;
  mindist = INT_MAX;

  per = contour_size / 2;
  for (i = 0; i < per; ++i) {
    dist = GeoDilationBin(bin, contour[i], contour[per + i]);
    if (dist > maxdist)
      maxdist = dist;
    if (dist < mindist)
      mindist = dist;
  }

  free(contour);

  *max = maxdist;
  *min = mindist;

}

// A new Approach to Compute Contour Salience Points with their
// Salience Values.

Curve3D *ContourSaliencePoints(Image *bin, float thres)
{
  int       R2=10*10,i1,i2,i3,i4;
  Image    *label=LabelContPixel(bin);
  AdjRel   *A=Circular(1.5);
  int       i,tmp,p,q,n=bin->ncols*bin->nrows;
  int       npts,*extsal,*intsal,nsals;
  int       *extsal1,*intsal1,*pts;
  Pixel     u,v,r;
  Image    *cost=CreateImage(bin->ncols,bin->nrows);
  Image    *root=CreateImage(bin->ncols,bin->nrows);
  GQueue   *Q=CreateGQueue(QSIZE,n,cost->val);
  Curve3D  *salience=NULL;
  float     theta;

  // initialization for the Euclidean IFT

  npts    = MaximumValue(label);
  pts     = AllocIntArray(npts+1);

  for (p=0; p < n; p++) {
    if (label->val[p] > 0){ // is seed
      cost->val[p]=0; 
      root->val[p]=p;
      pts[label->val[p]]=p;
      InsertGQueue(&Q,p);
    }else{
      root->val[p]=NIL;
      cost->val[p]=INT_MAX;
    }
  }

  // IFT up to a given radius.


  while(!EmptyGQueue(Q)){

    p   = RemoveGQueue(Q);

    u.x = p%cost->ncols;
    u.y = p/cost->ncols;
    r.x = root->val[p]%cost->ncols;
    r.y = root->val[p]/cost->ncols;
    for (i=1; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(cost,v.x,v.y)){
	q = v.x + cost->tbrow[v.y];
	if (cost->val[q] > cost->val[p]){
	  tmp = (v.x-r.x)*(v.x-r.x) + (v.y-r.y)*(v.y-r.y);
	  if ((tmp < cost->val[q])&&(tmp<=R2)){
	    if (cost->val[q]!=INT_MAX)
	      RemoveGQueueElem(Q,q);
	    cost->val[q]  = tmp;
	    root->val[q]  = root->val[p];
	    label->val[q] = label->val[p];
	    InsertGQueue(&Q,q);
	  }
	}
      }
    }
  }

  DestroyGQueue(&Q);

  /* Eliminate tie nodes */
  
  for (p=0; p < n; p++) {
    u.x = p%cost->ncols;
    u.y = p/cost->ncols;
    r.x = root->val[p]%cost->ncols;
    r.y = root->val[p]/cost->ncols;
    for (i=1; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(cost,v.x,v.y)){
	q = v.x + cost->tbrow[v.y];
	if (root->val[p]!=root->val[q]){
	  tmp = (v.x-r.x)*(v.x-r.x) + (v.y-r.y)*(v.y-r.y);
	  if (tmp == cost->val[q]) // q is a tie node
	    label->val[q]=0;
	}
      }
    }
  }
  

  DestroyAdjRel(&A);
  DestroyImage(&cost);

  // Copy contour points to the salience curve and compute their
  // salience values
  
  extsal  = AllocIntArray(npts+1);
  intsal  = AllocIntArray(npts+1);
  for (p=0; p < n; p++) {
    if (label->val[p]!=0){
      if (root->val[p]!=p) {
	if (bin->val[p]==0){ 
	  extsal[label->val[p]]++;
	}else{
	  intsal[label->val[p]]++;
	}
      }
    }
  }

  /* Merge relevant influence zones */

  extsal1 = AllocIntArray(npts+1);
  intsal1 = AllocIntArray(npts+1);

  for (i=1; i <= npts; i++) {
    i1 = (i-2);
    if (i1 < 1) i1=npts+i1; 
    i2 = (i+2);
    if (i2>npts) i2=i2%npts;
    i3 = (i-1);
    if (i3 < 1) i3=npts; 
    i4 = (i+1);
    if (i4>npts) i4=i4%npts;
    if (intsal[i]>extsal[i]){
      if((intsal[i]>=intsal[i1])&& 
	 (intsal[i]>=intsal[i2])&&
	 (intsal[i]>=intsal[i3])&& 
	 (intsal[i]>=intsal[i4])){
	intsal1[i]=intsal[i]+intsal[i1]+intsal[i2]+intsal[i3]+intsal[i4];      
      }
    }else{
      if (intsal[i]<extsal[i]){
	if ((extsal[i]>=extsal[i1])&& 
	    (extsal[i]>=extsal[i2])&&
	    (extsal[i]>=extsal[i3])&& 
	    (extsal[i]>=extsal[i4])){
	  extsal1[i]=extsal[i]+extsal[i1]+extsal[i2]+extsal[i3]+extsal[i4];
	}
      }
    }
  }

  // Eliminate non-salience points
  
  for (i=1; i <= npts; i++) {
    if (intsal1[i]<extsal1[i]){
      theta = (((float)2.0*extsal1[i])/((float)R2))*180.0/PI;
      if (theta<thres){	  	  
	pts[i]=NIL;
	intsal1[i]=extsal1[i]=0;
      }
    }else{
      if (intsal1[i]>extsal1[i]){
	theta = (((float)2.0*intsal1[i])/((float)R2))*180.0/PI;
	if (theta<thres){	  
	  pts[i]=NIL;
	  intsal1[i]=extsal1[i]=0;
	}
      }else{
	intsal1[i]=extsal1[i]=0;	
	pts[i]=NIL;
      }
    }
  }

  // Eliminate one from two consecutive saliences

  for (i=1; i <= npts; i++) 
    intsal[i]=extsal[i]=0;

  for (i=1; i <= npts; i++) {
    p = pts[i];
    if (p!=NIL){
      if (intsal1[i]<extsal1[i]){
	i1 = i-1;
	if (i1==0) i1=npts;
	q = pts[i1];
	if (q!=NIL){
	  if (intsal1[i1]<extsal1[i1]){
	    pts[i1]=NIL;
	  }
	}
      }else{
	if (intsal1[i]>extsal1[i]){
	  i1 = i-1;
	  if (i1==0) i1=npts;
	  q = pts[i1];
	  if (q!=NIL){
	    if (intsal1[i1]>extsal1[i1]){
	      pts[i1]=NIL;
	    }
	  }
	}
      }
    }
  }

  // Compute salience curve

  nsals=0;
  for (i=1; i <= npts; i++) 
    if (pts[i]!=NIL)
      nsals++;

  salience = CreateCurve3D(nsals);

  q=0;
  for (i=1; i <= npts; i++) {
    p = pts[i];
    if (p!=NIL){
      if (intsal1[i]<extsal1[i]){
	salience->X[q]=p%label->ncols;
	salience->Y[q]=p/label->ncols;
	salience->Z[q]=extsal1[i];	  
	printf("convexo %d (%d,%d) %d\n",i,p%label->ncols,p/label->ncols,extsal1[i]);
	q++;
      }else{
	if (intsal1[i]>extsal1[i]){
	  salience->X[q]=p%label->ncols;
	  salience->Y[q]=p/label->ncols;
	  salience->Z[q]=-intsal1[i];
	  printf("concavo %d (%d,%d) %d\n",i,p%label->ncols,p/label->ncols,-intsal1[i]);
	  q++;
	}
      }
    } 
  }
  free(pts);
  free(intsal);
  free(extsal);
  free(intsal1);
  free(extsal1);
  DestroyImage(&root);
  DestroyImage(&label);

  return(salience);
}
