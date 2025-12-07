#include "segmentation.h"
#include "border.h"
#include "set.h"
#include "queue.h"
#include "curve.h"
#include "radiometric.h"
#include "analysis.h"
#include "mathematics.h"
#include "geometry.h"
#include "morphology.h"
#include "scene.h"
#include "ctree.h"

Image   *RegionBorder(Image *label)
{
  Image *border=CreateImage(label->ncols,label->nrows);
  int p,q,i;
  Pixel u,v;
  AdjRel *A=Circular(1.0);

  for (u.y=0; u.y < label->nrows; u.y++) 
    for (u.x=0; u.x < label->ncols; u.x++) {
      p = u.x + label->tbrow[u.y];
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(label,v.x,v.y)){
	  q = v.x + label->tbrow[v.y];
	  if (label->val[q]!=label->val[p]){
	    border->val[p]=1;
	    break;
	  }
	}
      }
    }

  DestroyAdjRel(&A);
  return(border);
}

Image *EdgeDetection(Image *img)
{
  Image *grad,*label,*marker,*border;
  AdjRel *A=Circular(1.5);
  
  // Detect edges to avoid samples on edges

  grad   = MorphGrad(img,A);
  marker = CTVolumeClose(grad,(int)(0.00001*MaximumValue(img)*img->ncols*img->nrows));
  label  = WaterGray(grad,marker,A);
  border = RegionBorder(label);
  DestroyImage(&grad);
  DestroyImage(&marker);
  DestroyImage(&label);
  DestroyAdjRel(&A);

  return(border);
}

Image *GetRoot(AnnImg *aimg)
{
  Image *root=NULL;
  int p,n;

  root = CreateImage(aimg->img->ncols,aimg->img->nrows);
  n    = aimg->img->ncols*aimg->img->nrows;
  for (p=0; p < n; p++)
    if (aimg->pred->val[p]==p)
      root->val[p]=1;
    else
      root->val[p]=0;
  return(root);
}

Image *GetSKIZ(AnnImg *aimg)
{
  return(CompSKIZ(aimg));
}

Image *GetMarker(AnnImg *aimg)
{
  Image *root=NULL,*skiz=NULL,*lroot=NULL,*lskiz=NULL,*marker=NULL;
  AdjRel *A;

  A      = Circular(1.5);
  root   = GetRoot(aimg);
  lroot  = LabelBinComp(root,A);
  DestroyAdjRel(&A);
  skiz   = GetSKIZ(aimg);
  lskiz  = Paint(skiz,MaximumValue(lroot)+1);
  marker = Or(lroot,lskiz);
  DestroyImage(&root);
  DestroyImage(&lroot);
  DestroyImage(&skiz);
  DestroyImage(&lskiz);

  return(marker);
}

bool ValidContPoint(Image *bin, AdjRel *L, AdjRel *R, int p)
{
  int i,q,n,left,right;
  Pixel u,v,l,r;
  bool found=false;

  u.x = p%bin->ncols;
  u.y = p/bin->ncols;
  n   = L->n;

  for (i=0; i < n; i++) {
    v.x = u.x + L->dx[i];
    v.y = u.y + L->dy[i];
    if (ValidPixel(bin,v.x,v.y)){
      q = v.x + bin->tbrow[v.y];
      if ((bin->val[q]==1)&&(p!=q)){
	l.x = u.x + L->dx[i];
	l.y = u.y + L->dy[i];
	r.x = u.x + R->dx[i];
	r.y = u.y + R->dy[i];
	if (ValidPixel(bin,l.x,l.y))
	  left = l.x + bin->tbrow[l.y];
	else
	  left = -1;
	if (ValidPixel(bin,r.x,r.y))
	  right = r.x + bin->tbrow[r.y];
	else
	  right = -1;
	if (((left!=-1)&&(right!=-1)&&(bin->val[left]!=bin->val[right]))||
	    ((left==-1)&&(right!=-1)&&(bin->val[right]==1)) ||
	    ((right==-1)&&(left!=-1)&&(bin->val[left]==1))){
	  found = true;
	  break;
	}
      }
    }
  }

  return(found);
}


Image *LabelContour(Image *bin)
{
  Image *bndr=NULL;
  Image *color=NULL,*pred=NULL,*label=NULL;
  int p=0,q,r,i,j,left=0,right=0,n,*LIFO,last,l=1;
  AdjRel *A,*L,*R;
  Pixel u,v,w;

  A     = Circular(1.0);
  n     = bin->ncols*bin->nrows;
  bndr  = CreateImage(bin->ncols,bin->nrows);
  for (p=0; p < n; p++){
    if (bin->val[p]==1){
      u.x = p%bin->ncols;
      u.y = p/bin->ncols;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(bin,v.x,v.y)){
	  q = v.x + bin->tbrow[v.y];
	  if (bin->val[q]==0){
	    bndr->val[p]=1;
	    break;
	  }
	} else {
	    bndr->val[p]=1;
	    break;
	}
      }
    }
  }
  DestroyAdjRel(&A);

  A      = Circular(1.5);
  L      = LeftSide(A);
  R      = RightSide(A);
  label  = CreateImage(bndr->ncols,bndr->nrows);
  color  = CreateImage(bndr->ncols,bndr->nrows);
  pred   = CreateImage(bndr->ncols,bndr->nrows);
  LIFO   = AllocIntArray(n);
  last   = NIL;
  for (j=0; j < n; j++){
    if ((bndr->val[j]==1)&&
	(color->val[j]!=BLACK)&&
	ValidContPoint(bin,L,R,j)){
      last++; LIFO[last]    = j;
      color->val[j] = GRAY;
      pred->val[j] = j;
      while(last != NIL){
	p = LIFO[last];	last--;
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
		 ((right==-1)&&(left!=-1)&&(bin->val[left]==1))) ) {
	      pred->val[q] = p;
	      if (color->val[q] == WHITE){
		last++; LIFO[last] = q;
		color->val[q]=GRAY;
	      }
	    }
	  }
	}
      }
      r = p;
      while(pred->val[p]!=p){
	label->val[p] = l;
	p = pred->val[p];
      }
      if (r != p){
	label->val[p] = l;
	l++;
      }
    }
  }

  DestroyAdjRel(&A);
  DestroyAdjRel(&L);
  DestroyAdjRel(&R);
  DestroyImage(&bndr);
  DestroyImage(&color);
  DestroyImage(&pred);
  free(LIFO);
  return(label);
}


Image *LabelContPixel(Image *bin)
{
  Image *bndr=NULL;
  Image *color=NULL,*pred=NULL,*label=NULL;
  int p=0,q,r,i,j,n,left=0,right=0,*LIFO,last,l;
  AdjRel *A,*L,*R;
  Pixel u,v,w;

  A     = Circular(1.0);
  n     = bin->ncols*bin->nrows;
  bndr  = CreateImage(bin->ncols,bin->nrows);
  for (p=0; p < n; p++){
    if (bin->val[p]==1){
      u.x = p%bin->ncols;
      u.y = p/bin->ncols;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(bin,v.x,v.y)){
	  q = v.x + bin->tbrow[v.y];
	  if (bin->val[q]==0){
	    bndr->val[p]=1;
	    break;
	  }
	} else {
	    bndr->val[p]=1;
	    break;
	}
      }
    }
  }
  DestroyAdjRel(&A);

  A      = Circular(1.5);
  L      = LeftSide(A);
  R      = RightSide(A);
  label  = CreateImage(bndr->ncols,bndr->nrows);
  color  = CreateImage(bndr->ncols,bndr->nrows);
  pred   = CreateImage(bndr->ncols,bndr->nrows);
  n      = bndr->ncols*bndr->nrows;
  LIFO   = AllocIntArray(n);
  last   = NIL;
  for (j=0; j < n; j++){
    if ((bndr->val[j]==1)
	&&(color->val[j]!=BLACK)
	&&ValidContPoint(bin,L,R,j)){
      last++;
      LIFO[last]    = j;
      color->val[j] = GRAY;
      pred->val[j] = j;
      while(last != NIL){
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
		 ((right==-1)&&(left!=-1)&&(bin->val[left]==1)))){
	      pred->val[q] = p;
	      if (color->val[q] == WHITE){
		last++;
		LIFO[last] = q;
		color->val[q]=GRAY;
	      }
	    }
	  }
	}
      }
      r = p;
      l = 1;
      while(pred->val[p]!=p){
	label->val[p] = l;
	p = pred->val[p];
	l++;
      }
      if (r != p) {
	label->val[p] = l;
      }
    }
  }

  DestroyAdjRel(&A);
  DestroyAdjRel(&L);
  DestroyAdjRel(&R);
  DestroyImage(&bndr);
  DestroyImage(&color);
  DestroyImage(&pred);
  free(LIFO);
  return(label);
}

// 0 < thres < 1

Image *MSLabelComp(Image *img, AdjRel *A, float thres, int nfeats)
{
  Image *label=NULL;
  int i,j,n,p,q,l=1;
  int *FIFO=NULL;
  int first=0,last=0;
  float **feat=(float **)calloc(nfeats,sizeof(float *)),dmax,dist;
  int s,Imax;
  Image *img1,*img2;
  AdjRel *B;
  Pixel u,v;

  /* Compute multiscale features with nfeats scales */

  n = img->ncols*img->nrows;

  for (i=0; i < nfeats; i++)
    feat[i]=AllocFloatArray(n);

  Imax  = MaximumValue(img);
  img1  = ift_CopyImage(img);

  for (s=1; s <= nfeats; s=s+1) {
    B  = Circular(s);
    img2 = AsfOCRec(img1,B);
    for (i=0; i < n; i++) {
      feat[s-1][i] = (float)img2->val[i]/(float)Imax;
    }
    DestroyImage(&img2);
    DestroyAdjRel(&B);
  }
  DestroyImage(&img1);

  /* Compute dmax in the graph */

  dmax = 0.0;
  for (p=0; p < n; p++) {
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	dist = 0.0;
	for (s=1; s <= nfeats; s++){
	  dist += (feat[s-1][q]-feat[s-1][p])*(feat[s-1][q]-feat[s-1][p]);
	}
	if (dist > dmax)
	  dmax = dist;
      }
    }
  }
  dmax = sqrt(dmax);

  /* Label components */

  label = CreateImage(img->ncols,img->nrows);
  FIFO   = AllocIntArray(n);
  for (j=0; j < n; j++){
    if (label->val[j]==0){
      label->val[j]=l;
      FIFO[last]=j;
      last++;
      while(first != last){
	p = FIFO[first];
	u.x = p%img->ncols;
	u.y = p/img->ncols;
	first++;
	for (i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  if (ValidPixel(img,v.x,v.y)){
	    q = v.x + img->tbrow[v.y];
	    if (label->val[q]==0){
	      dist = 0.0;
	      for (s=1; s <= nfeats; s++){
		dist += (feat[s-1][q]-feat[s-1][p])*(feat[s-1][q]-feat[s-1][p]);	      }
	      if (sqrt(dist) <= thres*dmax) {
		label->val[q] = label->val[p];
		FIFO[last] = q;
		last++;
	      }
	    }
	  }
	}
      }
      l++;
      first=last=0;
    }
  }

  free(FIFO);
  for (i=0; i < nfeats; i++)
    free(feat[i]);
  free(feat);
  return(label);
}

Image *LabelComp(Image *img, AdjRel *A, int thres)
{
  Image *label=NULL,*flabel=NULL,*fimg=NULL;
  int i,j,n,sz,p,q,l=1;
  AdjPxl *N=NULL;
  int *FIFO=NULL;
  int first=0,last=0;

  sz     = FrameSize(A);
  fimg   = AddFrame(img,sz,INT_MIN);
  flabel = CreateImage(fimg->ncols,fimg->nrows);
  N      = AdjPixels(fimg,A);
  n      = fimg->ncols*fimg->nrows;
  FIFO   = AllocIntArray(n);
  for (j=0; j < n; j++){
    if ((flabel->val[j]==0)&&(fimg->val[j]!=INT_MIN)){
      flabel->val[j]=l;
      FIFO[last]=j;
      last++;
      while(first != last){
	p = FIFO[first];
	first++;
	for (i=1; i < N->n; i++){
	  q = p + N->dp[i];
	  if ((fimg->val[q]!=INT_MIN)&&(fabs(fimg->val[q]-fimg->val[p])<=thres)&&(flabel->val[q] == 0)){
	    flabel->val[q] = flabel->val[p];
	    FIFO[last] = q;
	    last++;
	  }
	}
      }
      l++;
      first=last=0;
    }
  }

  label = RemFrame(flabel,sz);
  DestroyAdjPxl(&N);
  DestroyImage(&fimg);
  DestroyImage(&flabel);
  free(FIFO);

  return(label);
}

Image *LabelBinComp(Image *bin, AdjRel *A)
{
  Image *label=NULL,*flabel=NULL,*fbin=NULL;
  int i,j,n,sz,p,q,l=1;
  AdjPxl *N=NULL;
  int *FIFO=NULL;
  int first=0,last=0;

  sz  = FrameSize(A);
  fbin = AddFrame(bin,sz,INT_MIN);
  flabel = CreateImage(fbin->ncols,fbin->nrows);
  N  = AdjPixels(fbin,A);
  n  = fbin->ncols*fbin->nrows;
  FIFO  = AllocIntArray(n);
  for (j=0; j < n; j++){
    if ((fbin->val[j]==1)&&(flabel->val[j]==0)){
      flabel->val[j]=l;
      FIFO[last]=j;
      last++;
      while(first != last){
	p = FIFO[first];
	first++;
	for (i=1; i < N->n; i++){
	  q = p + N->dp[i];
	  if ((fbin->val[q]==1)&&(flabel->val[q] == 0)){
	    flabel->val[q] = flabel->val[p];
	    FIFO[last] = q;
	    last++;
	  }
	}
      }
      l++;
      first=last=0;
    }
  }

  label = RemFrame(flabel,sz);
  DestroyAdjPxl(&N);
  DestroyImage(&fbin);
  DestroyImage(&flabel);
  free(FIFO);

  return(label);
}

Image *Paint(Image *bin, int value)
{
  Image *img=NULL;
  int p,n;

  img = ift_CopyImage(bin);
  n   = img->ncols*img->nrows;
  for (p=0; p < n; p++)
    if (bin->val[p]==1)
      img->val[p]=value;
  return(img);
}

Image *Threshold(Image *img, int lower, int higher)
{
  Image *bin=NULL;
  int p,n;

  bin = CreateImage(img->ncols,img->nrows);
  n = img->ncols*img->nrows;
  for (p=0; p < n; p++)
    if ((img->val[p] >= lower)&&(img->val[p] <= higher))
      bin->val[p]=1;
  return(bin);
}

int AutoThreshold(Image *img)
{
  Curve *hist=NULL;
  int i,n,xmin = 0;
  long dd, d, d0, dmin;

  hist = Histogram(img);

  n = hist->n;
  dmin = d0 = 0;
  for (i = 1; i < n; i++) {
    d = hist->Y[i] - hist->Y[i-1];
    dd = d - d0;
    if (dmin > dd) {
      dmin = dd;
      xmin = i;
    }
    d0 = d;
  }

  DestroyCurve(&hist);
  return xmin;
}


int AcceptEdge(AnnImg *aimg, int *edge)
{
  int p,n;

  DestroySet(&(aimg->seed));
  n = aimg->img->ncols*aimg->img->nrows;
  for (p=0; p < n; p++) {
    aimg->cost->val[p]  = INT_MAX;
    aimg->label->val[p] = 0;
    aimg->pred->val[p]  = p;
  }

  n = edge[0];
  for (p=1; p < n; p++){
    aimg->cost->val[edge[p]]=INT_MIN;
    aimg->pred->val[edge[p]]=edge[p+1];
  }
  aimg->cost->val[edge[p]]=INT_MIN;
  aimg->pred->val[edge[p]]=edge[p];

  return(edge[1]);
}

Image *Clustering(Image *img, AdjRel *A, int Co, int Ao)
{
  Queue *Q=NULL;
  int i,p,q,r,s,tmp,Imax,n,l=1;
  Pixel u,v;
  Image *root,*cost,*label;
  Curve *hist;
  char *color=NULL;

  root = CreateImage(img->ncols,img->nrows);
  cost = CreateImage(img->ncols,img->nrows);
  n = img->ncols*img->nrows;
  color = AllocCharArray(n);
  Imax = MaximumValue(img);
  Q = CreateQueue(Imax+2,n);

  for (p=0; p < n; p++){
    root->val[p] = p;
    cost->val[p] = Imax+1;
    color[p]=GRAY;
    InsertQueue(Q,cost->val[p],p);
  }

  while (!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    color[p]=BLACK;
    r   = SeedComp(root,p); if (r == p) cost->val[r] = 0;
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (color[q] != BLACK){
	  s = SeedComp(root,q);
	  tmp = abs(img->val[r] - img->val[s]);
	  if ((tmp < cost->val[q])&&(tmp < Co)){
	    UpdateQueue(Q,q,cost->val[q],tmp);
	    if (Q->C.current > tmp)
	      Q->C.current = tmp;
	    cost->val[q] = tmp;
	    root->val[s] = r;
	  }
	}
      }
    }
  }

  free(color);
  ResetQueue(Q);

  for (p=0; p < n; p++){
    root->val[p] = SeedComp(root,p);
    cost->val[p] = INT_MAX;
  }

  hist  = Histogram(root);
  label = CreateImage(img->ncols,img->nrows);
  l = 1;
  for (i=0; i < hist->n; i++)
    if (hist->Y[i] >= Ao){
      p = (int)hist->X[i];
      cost->val[p] = 0;
      label->val[p] = l;
      l++;
      InsertQueue(Q,cost->val[p],p);
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
	if (cost->val[p] < cost->val[q]){
	  tmp = MAX(cost->val[p],abs(img->val[root->val[p]]-img->val[root->val[q]]));
	  if (tmp < cost->val[q]){
	    if (cost->val[q] == INT_MAX)
	      InsertQueue(Q,tmp,q);
	    else
	      UpdateQueue(Q,q,cost->val[q],tmp);
	    cost->val[q] = tmp;
	    label->val[q] = label->val[p];
	  }
	}
      }
    }
  }

  DestroyImage(&cost);
  DestroyImage(&root);
  DestroyQueue(&Q);
  DestroyCurve(&hist);

  return(label);
}


Image *Highlight(Image *img, Image *label, int value)
{
  Image *himg=NULL;
  int p,q,i;
  AdjRel *A=NULL;
  Pixel u,v;

  himg = ift_CopyImage(img);
  A    = Circular(1.0);
  for (u.y=0; u.y < himg->nrows; u.y++){
    for (u.x=0; u.x < himg->ncols; u.x++){
      p = u.x + himg->tbrow[u.y];
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(himg,v.x,v.y)){
	  q = v.x + himg->tbrow[v.y];
	  if (label->val[p] > label->val[q]){
	    himg->val[p] = value;
	    break;
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return(himg);
}

CImage *CHighlight(CImage *cimg, Image *label, float R, float G, float B)
{
  CImage *himg=NULL;
  int p,q,i;
  AdjRel *A=NULL;
  Pixel u,v;

  himg = CopyCImage(cimg);
  A    = Circular(1.0);
  for (u.y=0; u.y < himg->C[0]->nrows; u.y++){
    for (u.x=0; u.x < himg->C[0]->ncols; u.x++){
      p = u.x + himg->C[0]->tbrow[u.y];
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(himg->C[0],v.x,v.y)){
	  q = v.x + himg->C[0]->tbrow[v.y];
	  if (label->val[p] > label->val[q]){
	    himg->C[0]->val[p] = (int)(255.0*R);
	    himg->C[1]->val[p] = (int)(255.0*G);
	    himg->C[2]->val[p] = (int)(255.0*B);
	    break;
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return(himg);
}

CImage *CHighlightObject(CImage *cimg, Image *label, int obj, int value)
{
  CImage *himg=NULL;
  int p,q,i;
  AdjRel *A=NULL;
  Pixel u,v;

  himg = CopyCImage(cimg);
  A    = Circular(1.0);
  for (u.y=0; u.y < himg->C[0]->nrows; u.y++){
    for (u.x=0; u.x < himg->C[0]->ncols; u.x++){
      p = u.x + himg->C[0]->tbrow[u.y];
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(himg->C[0],v.x,v.y)){
	  q = v.x + himg->C[0]->tbrow[v.y];
	  if (label->val[p] > label->val[q]){
	    himg->C[0]->val[p] = value;
	    himg->C[1]->val[p] = value;
	    himg->C[2]->val[p] = value;
	    break;
	  }
	}
      }
    }
  }
  DestroyAdjRel(&A);

  return(himg);
}

int *iftFindEdge(AnnImg *aimg, AdjRel *A, int src, int dst)
{
  Queue *Q=NULL;
  int i,p,q,n,tmp,mincost=INT_MAX;
  Pixel u,v;
  bool closing=false;

  if (aimg->pred->val[dst]==dst) {

    if (aimg->cost->val[dst] == INT_MIN){    /* Closing Operation */
      aimg->cost->val[dst] = INT_MAX;
      closing = true;
    }

    n = aimg->img->ncols*aimg->img->nrows;
    Q = CreateQueue((int)pow(MaximumValue(aimg->img),1.5)+1,n);

    if (aimg->seed == NULL) {
      AddSeed(aimg, src, 0, 1, aimg->pred->val[src]);
    }
    while (aimg->seed != NULL){
      p=RemoveSet(&(aimg->seed));
      if (aimg->cost->val[p] < mincost)
	mincost = aimg->cost->val[p];
      InsertQueue(Q,aimg->cost->val[p]%Q->C.nbuckets,p);
    }
    Q->C.current = mincost%Q->C.nbuckets;

    while(!EmptyQueue(Q)) {
      p = RemoveQueue(Q);
      if (p==dst){
	InsertSet(&(aimg->seed),p);
	break;
      }
      u.x = p%aimg->img->ncols;
      u.y = p/aimg->img->ncols;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(aimg->img,v.x,v.y)){
	  q = v.x + aimg->img->tbrow[v.y];
	  if (aimg->cost->val[p] < aimg->cost->val[q]){
	    tmp = aimg->cost->val[p]+(int)pow(aimg->img->val[q],1.5);
	    if (tmp < aimg->cost->val[q]){
	      if (aimg->cost->val[q]==INT_MAX)
		InsertQueue(Q,tmp%Q->C.nbuckets,q);
	      else
		UpdateQueue(Q,q,aimg->cost->val[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	      aimg->cost->val[q]  = tmp;
	      aimg->pred->val[q]  = p;
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
  }

  if (closing){
    tmp = aimg->pred->val[dst];
    aimg->pred->val[dst] = dst;
    dst = tmp;
  }

  return(Path(aimg->pred,dst));
}

int *iftFindOEdge(AnnImg *aimg, AdjRel *A, Image *oimg, char orient, int src, int dst)
{
  Queue *Q=NULL;
  int i,p,q,n,tmp,left,right,Cmax,mincost=INT_MAX;
  Pixel u,v;
  bool closing=false;
  AdjRel *L,*R;

  if (aimg->pred->val[dst]==dst) {

    if (aimg->cost->val[dst] == INT_MIN){    /* Closing Operation */
      aimg->cost->val[dst] = INT_MAX;
      closing = true;
    }

    Cmax = (int)pow(MaximumValue(aimg->img),1.5);
    n = aimg->img->ncols*aimg->img->nrows;
    Q = CreateQueue(Cmax+1,n);

    if (aimg->seed == NULL) {
      AddSeed(aimg, src, aimg->img->val[src], 1, aimg->pred->val[src]);
    }

    while (aimg->seed != NULL){
      p=RemoveSet(&(aimg->seed));
      if (aimg->cost->val[p] < mincost)
	mincost = aimg->cost->val[p];
      InsertQueue(Q,aimg->cost->val[p]%Q->C.nbuckets,p);
    }
    Q->C.current = mincost%Q->C.nbuckets;

    L = LeftSide(A);
    R = RightSide(A);

    while(!EmptyQueue(Q)) {
      p = RemoveQueue(Q);
      if (p==dst){
	InsertSet(&(aimg->seed),p);
	break;
      }
      u.x = p%aimg->img->ncols;
      u.y = p/aimg->img->ncols;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(aimg->img,v.x,v.y)){
	  q = v.x + aimg->img->tbrow[v.y];
	  if (aimg->cost->val[p] < aimg->cost->val[q]){
	    v.x = u.x + L->dx[i];
	    v.y = u.y + L->dy[i];
	    if (ValidPixel(aimg->img,v.x,v.y))
	      left = v.x + aimg->img->tbrow[v.y];
	    else
	      left = q;
	    v.x = u.x + R->dx[i];
	    v.y = u.y + R->dy[i];
	    if (ValidPixel(aimg->img,v.x,v.y))
	      right = v.x + aimg->img->tbrow[v.y];
	    else
	      right = q;
	    if (orient == 1){
	      if (oimg->val[left] < oimg->val[right])
		tmp = aimg->cost->val[p]+(int)pow(aimg->img->val[q],1.5);
	      else
		tmp = aimg->cost->val[p]+Cmax;
	    }else{  /* assuming that orient is 2 */
	      if (oimg->val[left] > oimg->val[right])
		tmp = aimg->cost->val[p]+(int)pow(aimg->img->val[q],1.5);
	      else
		tmp = aimg->cost->val[p]+Cmax;
	    }
	    if (tmp < aimg->cost->val[q]){
	      if (aimg->cost->val[q] == INT_MAX)
		InsertQueue(Q,tmp%Q->C.nbuckets,q);
	      else
		UpdateQueue(Q,q,aimg->cost->val[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	      aimg->cost->val[q]  = tmp;
	      aimg->pred->val[q]  = p;
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
    DestroyAdjRel(&L);
    DestroyAdjRel(&R);
  }

  if (closing){
    tmp = aimg->pred->val[dst];
    aimg->pred->val[dst] = dst;
    dst = tmp;
  }

  return(Path(aimg->pred,dst));
}

Image *RegMin(Image *img, AdjRel *A)
{
  Image *fcost=NULL,*fimg=NULL,*fpred=NULL,*frmin=NULL,*rmin=NULL;
  Queue *Q=NULL;
  int i,sz,p,q,n;
  AdjPxl *N=NULL;

  sz = FrameSize(A);
  fcost = AddFrame(img,sz,INT_MIN);
  fimg = AddFrame(img,sz,INT_MIN);
  frmin = CreateImage(fcost->ncols,fcost->nrows);
  fpred = CreateImage(fcost->ncols,fcost->nrows);
  N  = AdjPixels(fcost,A);
  n = fcost->ncols*fcost->nrows;
  Q = CreateQueue(MaximumValue(img)+1,n);

  for(p = 0; p < n; p++)
    if (fcost->val[p] != INT_MIN){
      InsertQueue(Q,fcost->val[p],p);
      fpred->val[p]=p;
    }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (fpred->val[p]==p)
      frmin->val[p]=1;
    else
      frmin->val[p]=0;

    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] < fcost->val[q])
	if (fimg->val[p] <= fimg->val[q]){
	  UpdateQueue(Q,q,fcost->val[q],fcost->val[p]);
	  fcost->val[q] = fcost->val[p];
	  fpred->val[q] = p;
	}
    }
  }


  rmin  = RemFrame(frmin,sz);

  DestroyQueue(&Q);
  DestroyImage(&fimg);
  DestroyImage(&fcost);
  DestroyImage(&fpred);
  DestroyImage(&frmin);
  DestroyAdjPxl(&N);

  return(rmin);
}

Image *RegMax(Image *img, AdjRel *A)
{
  Image *fcost=NULL,*fimg=NULL,*fpred=NULL,*frmax=NULL,*rmax=NULL;
  Queue *Q=NULL;
  int i,sz,p,q,n,Imax;
  AdjPxl *N=NULL;

  sz = FrameSize(A);
  fcost = AddFrame(img,sz,INT_MIN);
  fimg = AddFrame(img,sz,INT_MIN);
  frmax = CreateImage(fcost->ncols,fcost->nrows);
  fpred = CreateImage(fcost->ncols,fcost->nrows);
  N  = AdjPixels(fcost,A);
  n = fcost->ncols*fcost->nrows;
  Imax = MaximumValue(img);
  Q = CreateQueue(Imax+1,n);

  for(p = 0; p < n; p++)
    if (fcost->val[p] != INT_MIN){
      fcost->val[p] = Imax - fcost->val[p];
      InsertQueue(Q,fcost->val[p],p);
      fpred->val[p]=p;
    }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (fpred->val[p]==p)
      frmax->val[p]=1;
    else
      frmax->val[p]=0;

    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] < fcost->val[q])
	if (fimg->val[p] >= fimg->val[q]){
	  UpdateQueue(Q,q,fcost->val[q],fcost->val[p]);
	  fcost->val[q] = fcost->val[p];
	  fpred->val[q] = p;
	}
    }
  }


  rmax  = RemFrame(frmax,sz);

  DestroyQueue(&Q);
  DestroyImage(&fimg);
  DestroyImage(&fcost);
  DestroyImage(&fpred);
  DestroyImage(&frmax);
  DestroyAdjPxl(&N);

  return(rmax);
}

Image *HDomes(Image *img, AdjRel *A, int H)
{
  Image *himg=NULL,*infrec=NULL,*diff=NULL,*hdomes=NULL;

  himg   = Sub(img,H);
  infrec = InfRec(img,himg,A);
  diff   = Diff(img,infrec);
  hdomes = Threshold(diff,1,INT_MAX);
  DestroyImage(&himg);
  DestroyImage(&infrec);
  DestroyImage(&diff);

  return(hdomes);
}

Image *HBasins(Image *img, AdjRel *A, int H)
{
  Image *himg=NULL,*suprec=NULL,*diff=NULL,*hbasins=NULL;

  himg    = Add(img,H);
  suprec  = SupRec(img,himg,A);
  diff    = Diff(suprec,img);
  hbasins = Threshold(diff,1,INT_MAX);
  DestroyImage(&himg);
  DestroyImage(&suprec);
  DestroyImage(&diff);

  return(hbasins);
}

Image *SelectHDomes(Image *img, Image *seed, AdjRel *A, int H)
{
  Image *cost=NULL,*label=NULL;
  Queue *Q=NULL;
  int i,p,q,Imax,n;
  Pixel u,v;

  Imax  = MaximumValue(img);
  n     = img->ncols*img->nrows;
  cost  = ift_CopyImage(img);
  label = ift_CopyImage(seed);
  Q    = CreateQueue(Imax+1,n);
  for (p=0; p < n; p++){
    if (seed->val[p] > 0) {
      cost->val[p] = MAX(img->val[p]-H,0);
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
	if (cost->val[p] < cost->val[q])
	  {
	    if (Q->L.elem[q].color==GRAY)
	      UpdateQueue(Q,cost->val[q],cost->val[p],q);
	    else
	      InsertQueue(Q,cost->val[p],q);
	    cost->val[q]  = cost->val[p];
	    label->val[q] = label->val[p];
	  }
      }
    }
  }

  DestroyQueue(&Q);
  DestroyImage(&cost);

  return(label);
}

Image *SelectHBasins(Image *img, Image *seed, AdjRel *A, int H)
{
  Image *cost=NULL,*label=NULL;
  Queue *Q=NULL;
  int i,p,q,Imax,n;
  Pixel u,v;

  Imax  = MaximumValue(img);
  n     = img->ncols*img->nrows;
  cost  = ift_CopyImage(img);
  label = ift_CopyImage(seed);
  Q    = CreateQueue(Imax+1,n);
  for (p=0; p < n; p++){
    if (seed->val[p] > 0) {
      cost->val[p] = MIN(img->val[p]+H,Imax);
      InsertQueue(Q,Imax-cost->val[p],p);
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
	    if (Q->L.elem[q].color==GRAY)
	      UpdateQueue(Q,Imax-cost->val[q],Imax-cost->val[p],q);
	    else
	      InsertQueue(Q,Imax-cost->val[p],q);
	    cost->val[q]  = cost->val[p];
	    label->val[q] = label->val[p];
	  }
      }
    }
  }

  DestroyQueue(&Q);
  DestroyImage(&cost);

  return(label);
}

/*Image *iftWatershed(AnnImg *aimg, AdjRel *A)
{
  iftBasins(aimg, A);
  return(ift_CopyImage(aimg->label));
}*/

Image *IncWater(AnnImg *aimg, AdjRel *A)
{
  Queue *Q=NULL;
  int s,t,tmp,i;
  Pixel u,v;

  Q = CreateQueue(MaximumValue(aimg->img)+1,aimg->img->ncols*aimg->img->nrows);

  while(aimg->seed != NULL) {
    t=RemoveSet(&(aimg->seed));
    InsertQueue(Q,aimg->cost->val[t],t);
  }

  while (!EmptyQueue(Q)) {
    s=RemoveQueue(Q);
    u.x = s%aimg->img->ncols;
    u.y = s/aimg->img->ncols;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	t = v.x + aimg->img->tbrow[v.y];
	if ((aimg->cost->val[s] < aimg->cost->val[t])||
	    (s == aimg->pred->val[t])){
	  tmp = MAX(aimg->cost->val[s],aimg->img->val[t]);
	  if ((tmp < aimg->cost->val[t])||
	      (s == aimg->pred->val[t])){
	    InsertQueue(Q,tmp,t);
	    aimg->cost->val[t]  = tmp;
	    aimg->pred->val[t]  = s;
	    aimg->label->val[t] = aimg->label->val[s];
	  }
	}
      }
    }
  }

  DestroyQueue(&Q);

  return(ift_CopyImage(aimg->label));
}

Image *DecWater(AnnImg *aimg, AdjRel *A)
{
  Queue *Q=NULL;
  int s,t,tmp,i;
  Voxel u,v;

  Q = CreateQueue(MaximumValue(aimg->img)+1,aimg->img->ncols*aimg->img->nrows);

  while(aimg->seed != NULL) {
    t=Seed(aimg->pred,RemoveSet(&(aimg->seed)));
    if (aimg->cost->val[t] != INT_MAX){
      InsertQueue(Q,0,t);
      aimg->cost->val[t]  = INT_MAX;
      aimg->pred->val[t]  = t;
      aimg->label->val[t] = 0;
    }
  }

  while (!EmptyQueue(Q)) {
    s=RemoveQueue(Q);
    u.x = s%aimg->img->ncols;
    u.y = s/aimg->img->ncols;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	t = v.x + aimg->img->tbrow[v.y];
	if (aimg->pred->val[t] == s){
	  aimg->cost->val[t] = INT_MAX;
	  aimg->pred->val[t] = t;
	  aimg->label->val[t] = 0;
	  InsertQueue(Q,0,t);
	}else
	  if (aimg->pred->val[t] != t)
	    InsertSet(&(aimg->seed),t);
      }
    }
  }

  while(aimg->seed != NULL) {
    t=RemoveSet(&(aimg->seed));
    if (aimg->pred->val[t] != t)
      InsertQueue(Q,aimg->cost->val[t],t);
  }

  Q->C.current = 0;

  while (!EmptyQueue(Q)) {
    s=RemoveQueue(Q);
    u.x = s%aimg->img->ncols;
    u.y = s/aimg->img->ncols;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(aimg->img,v.x,v.y)){
	t = v.x + aimg->img->tbrow[v.y];
	if (aimg->cost->val[s] < aimg->cost->val[t]){
	  tmp = MAX(aimg->cost->val[s],aimg->img->val[t]);
	  if (tmp < aimg->cost->val[t]){
	    InsertQueue(Q,tmp,t);
	    aimg->cost->val[t] = tmp;
	    aimg->pred->val[t] = s;
	    aimg->label->val[t] = aimg->label->val[s];
	  }
	}
      }
    }
  }

  DestroyQueue(&Q);
  return(ift_CopyImage(aimg->label));
}

Image *WaterLabel(Image *img, Image *label, AdjRel *A)
{
  Image *water=NULL,*fimg=NULL,*flabel=NULL,*fcost=NULL;
  Queue *Q=NULL;
  int i,sz,p,q,tmp,n;
  AdjPxl *N=NULL;

  sz = FrameSize(A);
  fimg = AddFrame(img,sz,INT_MIN);
  flabel = AddFrame(label,sz,INT_MIN);
  fcost = CreateImage(fimg->ncols,fimg->nrows);
  N  = AdjPixels(fcost,A);
  n  = fcost->ncols*fcost->nrows;
  Q = CreateQueue(MaximumValue(img)+1,n);

  for (p=0; p < n; p++)
    if (flabel->val[p] >= 0){
      fcost->val[p]=0; /* Guarantee minima imposition */
      InsertQueue(Q,fcost->val[p],p);
    } else {
      if (flabel->val[p] != INT_MIN)
	fcost->val[p]=INT_MAX;
      else{
	fcost->val[p]=fimg->val[p];
      }
    }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] < fcost->val[q]){
	tmp = MAX(fcost->val[p],fimg->val[q]);
	if (tmp < fcost->val[q]){
	  InsertQueue(Q,tmp,q);
	  fcost->val[q] = tmp;
	  flabel->val[q] = flabel->val[p];
	}
      }
    }
  }

  water = RemFrame(flabel,sz);

  DestroyQueue(&Q);
  DestroyAdjPxl(&N);
  DestroyImage(&fimg);
  DestroyImage(&flabel);
  DestroyImage(&fcost);

  return(water);
}

Image *WaterBin(Image *img, Image *bin, AdjRel *A)
{
  Image *water=NULL,*fbin=NULL,*fimg=NULL,*flabel=NULL,*fcost=NULL;
  Queue *Q=NULL;
  int i,sz,p,q,tmp,n,r=1;
  AdjPxl *N=NULL;

  sz = FrameSize(A);
  fimg = AddFrame(img,sz,INT_MIN);
  fbin = AddFrame(bin,sz,INT_MIN);
  fcost  = CreateImage(fimg->ncols,fimg->nrows);
  flabel = CreateImage(fimg->ncols,fimg->nrows);
  N  = AdjPixels(fcost,A);
  n  = fcost->ncols*fcost->nrows;
  Q = CreateQueue(MaximumValue(img)+1,n);

  for (p=0; p < n; p++)
    if (fbin->val[p] > 0){
      fcost->val[p]=1;
      fimg->val[p] =0;
      InsertQueue(Q,fcost->val[p],p);
    } else {
      if (fbin->val[p] != INT_MIN){
	fcost->val[p]=INT_MAX;
	fimg->val[p] =MAX(fimg->val[p],1);
      }else{
	fcost->val[p]=fimg->val[p];
      }
    }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (flabel->val[p]==0){
      flabel->val[p] = r;
      r++;
      fcost->val[p] = 0;  /* Guarantee minima imposition */
      Q->C.current  = 0;
    }
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->val[p] < fcost->val[q]){
	tmp = MAX(fcost->val[p],fimg->val[q]);
	if (tmp < fcost->val[q]){
	  if (fbin->val[q] > 0)
	    RemoveQueueElem(Q,q,fcost->val[q]);
	  InsertQueue(Q,tmp,q);
	  fcost->val[q]  = tmp;
	  flabel->val[q] = flabel->val[p];
	}
      }
    }
  }

  water = RemFrame(flabel,sz);

  DestroyQueue(&Q);
  DestroyAdjPxl(&N);
  DestroyImage(&fimg);
  DestroyImage(&fbin);
  DestroyImage(&flabel);
  DestroyImage(&fcost);

  return(water);
}

Image *WaterGray(Image *img, Image *marker, AdjRel *A)
{
  Image *cost=NULL,*label=NULL;
  GQueue *Q=NULL;
  int i,p,q,tmp,n,r=1;
  Pixel u,v;

  n     = img->ncols*img->nrows;
  cost  = CreateImage(img->ncols,img->nrows);
  label = CreateImage(img->ncols,img->nrows);
  Q     = CreateGQueue(MaximumValue(marker)+2,n,cost->val);

  for (p=0; p < n; p++) {
    cost->val[p]=marker->val[p]+1;
    InsertGQueue(&Q,p);
  }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
    if (label->val[p]==0) {
      cost->val[p]=img->val[p];
      label->val[p]=r;
      r++;
    }
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (Q->L.elem[q].color != BLACK){
	  tmp = MAX(cost->val[p],img->val[q]);
	  if (tmp < cost->val[q]){
	    UpdateGQueue(&Q,q,tmp);
	    label->val[q] = label->val[p];
	  }
	}
      }
    }
  }

  DestroyGQueue(&Q);
  DestroyImage(&cost);

  return(label);
}

Image *WaterMax(Image *img, Image *marker, AdjRel *A)
{
  Image *cost=NULL,*label=NULL;
  Queue *Q=NULL;
  int i,p,q,tmp,n,r=1,thres;
  Pixel u,v;

  n     = img->ncols*img->nrows;
  cost  = CreateImage(img->ncols,img->nrows);
  label = CreateImage(img->ncols,img->nrows);
  Q     = CreateQueue(MaximumValue(marker)+2,n);
  thres = MaximumValue(img);
  for (p=0; p < n; p++)
    if (marker->val[p] < thres){
      cost->val[p]=marker->val[p]+1;
      InsertQueue(Q,cost->val[p],p);
    }else
      cost->val[p]=img->val[p];

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (label->val[p]==0) {
      cost->val[p]=marker->val[p];
      label->val[p]=r;
      r++;
      Q->C.current=marker->val[p];
    }
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	tmp = MAX(cost->val[p],img->val[q]);
	if (tmp < cost->val[q]){
	  UpdateQueue(Q,q,cost->val[q],tmp);
	  cost->val[q]  = tmp;
	  label->val[q] = label->val[p];
	}
      }
    }
  }

  DestroyQueue(&Q);
  DestroyImage(&cost);

  return(label);
}


int *CutPath(AnnImg *aimg, int src, int dst)
{
  /* Cuts the path in the annotated image from pixel src to the nearest pixel of dst */
  int i, j, k, n;
  int distsq, min_distsq = INT_MAX;
  Pixel p, p_nearest, ptmp;

  p.x = dst % aimg->pred->ncols;
  p.y = dst / aimg->pred->ncols;
  p_nearest = p;

  /* find the nearest point */
  for (i = src;;) {
    ptmp.x = i % aimg->pred->ncols;
    ptmp.y = i / aimg->pred->ncols;
    distsq = (p.x - ptmp.x)*(p.x - ptmp.x) + (p.y - ptmp.y)*(p.y - ptmp.y);
    if (distsq < min_distsq) {
      min_distsq = distsq;
      p_nearest = ptmp;
    }
    if (i == aimg->pred->val[i]) break;
    i = aimg->pred->val[i];
  }

  /* cut pixels */
  i = src;
  j = p_nearest.x + aimg->pred->tbrow[p_nearest.y];
  while (i != j) {
    aimg->cost->val[i] = INT_MAX;
    aimg->label->val[i] = 0;
    k = aimg->pred->val[i];
    aimg->pred->val[i] = i;
    i = k;
  }

  /* reset all pixels that have as root a cutted pixel */
  n = aimg->img->ncols * aimg->img->nrows;
  for (i=0; i<n; i++) {
    k = Seed(aimg->pred, i);
    if ((k != i) && (aimg->cost->val[k] == INT_MAX)) {
      aimg->cost->val[i] = INT_MAX;
      aimg->label->val[i] = 0;
      aimg->pred->val[i] = i;
    }
  }

  return (Path(aimg->pred, j));
}

Image *TreeComputing(Image *grad, Curve *pts)
{
  AnnImg *aimg=Annotate(grad,NULL,NULL);
  int p,i;
  AdjRel *A=Circular(1.0);
  Image *pred=NULL;

  for (i=0; i < pts->n; i++) {
    p = (int)pts->X[i] + grad->tbrow[(int)pts->Y[i]];
    if (ValidPixel(grad,(int)pts->X[i],(int)pts->Y[i]))
      AddSeed(aimg,p,0,1,p);
  }
  iftBasins(aimg,A);
  pred = ift_CopyImage(aimg->pred);
  DestroyAdjRel(&A);
  DeAnnotate(&aimg);

  return(pred);
}

Image *TreeCounting(Image *pred)
{
  int p,q,n;
  Image *nsons=NULL;

  n     = pred->ncols*pred->nrows;
  nsons = CreateImage(pred->ncols,pred->nrows);
  for (p=0; p < n; p++) {
    q = p;
    while(pred->val[q]!=q){
      nsons->val[q]++;
      q = pred->val[q];
    }
    nsons->val[q]++;
  }
  return(nsons);
}

void WriteNSons(Image *nsons, Image *pred)
{
  int p,q,n;
  FILE *fp;
  char filename[100];

  n     = pred->ncols*pred->nrows;
  for (p=0; p < n; p++) {
    q = p;
    sprintf(filename,"nsons%d.txt",p);
    fp = fopen(filename,"w");
    while(pred->val[q]!=q){
      fprintf(fp,"%d\n",nsons->val[q]);
      q = pred->val[q];
    }
    fprintf(fp,"%d\n",nsons->val[q]);
    fclose(fp);
  }
}

Image *TreePruning(Image *nsons, Image *pred, Curve *pts)
{
  int  p,q,i,n;
  Image *label;
  char *endpts=NULL;

  label  = CreateImage(pred->ncols,pred->nrows);
  n      = pred->ncols*pred->nrows;
  endpts = AllocCharArray(n);

  for (i=0; i < pts->n; i++){
    p = (int)pts->X[i] + pred->tbrow[(int)pts->Y[i]];
    endpts[p]=1;
  }

  for (p=0; p < n; p++) {
    q = p;
    while (pred->val[q] != q) {
      if (endpts[q]==1){
	goto NEXT;
      }
      q = pred->val[q];
    }
    if (endpts[q]==1)
      goto NEXT;
    q = p;
    while (pred->val[q] != q) {
      label->val[q]=1;
      q = pred->val[q];
    }
    label->val[q]=1;

  NEXT:
    ;
  }
  free(endpts);

  return(label);

}

int Otsu(Image *img)
{
  Curve *hist=NormHistogram(img);
  double p1,p2,m1,m2,s1,s2,J,Jmax=-1.0;
  int i,T,Topt=0,Imax=MaximumValue(img);

  for (T=1; T < Imax; T++){
    p1 = 0.0;
    for (i=0; i <= T; i++)
      p1 += hist->Y[i];
    p2 = 1.0 - p1;
    if ((p1 > 0.0)&&(p2 > 0.0)){
      m1 = 0.0;
      for (i=0; i <= T; i++)
	m1 += hist->Y[i]*i;
      m1 /= p1;
      m2 = 0.0;
      for (i=T+1; i <= Imax; i++)
	m2 += hist->Y[i]*i;
      m2 /= p2;
      s1 = 0.0;
      for (i=0; i <= T; i++)
	s1 += hist->Y[i]*(i-m1)*(i-m1);
      s1 /= p1;
      s2 = 0.0;
      for (i=T+1; i <= Imax; i++)
	s2 += hist->Y[i]*(i-m2)*(i-m2);
      s2 /= p2;
      J = (p1*p2*(m1-m2)*(m1-m2))/(p1*s1+p2*s2);
    }else{
      J = 0.0;
    }
    if (J > Jmax){
      Jmax = J;
      Topt = T;
    }
  }
  DestroyCurve(&hist);
  return(Topt);
}

Image *iftThres(Image *img, AdjRel *A)
{
  int     Th=0, Tl=0, Imax = MaximumValue(img);
  Image  *cost=NULL,*label=NULL,*root=NULL;
  GQueue *Q=NULL;
  int     p,q,i,tmp,n;
  Pixel   u,v;
  Curve   *hist=NormAccHistogram(img);

  for (i=hist->n-1; i > 0; i--){
    if (hist->Y[i] < 0.98){
      Th = i;
      break;
    }
  }
  for (i=0; i < hist->n; i++){
    if (hist->Y[i] > 0.02){
      Tl = i;
      break;
    }
  }
  printf("Tl %d Th %d\n",Tl,Th);
 
  n     = img->ncols*img->nrows;
  cost  = CreateImage(img->ncols,img->nrows);
  label = CreateImage(img->ncols,img->nrows);
  root  = CreateImage(img->ncols,img->nrows);
  Q     = CreateGQueue(Imax+1,n,cost->val);

  // Initialization

  for (p=0; p < n; p++) {
    label->val[p]=0; root->val[p]=p;	
    if ((img->val[p] > Tl)&&
	(img->val[p] < Th)){
      cost->val[p]=INT_MAX; 
    }else{
      cost->val[p]=0; 
      InsertGQueue(&Q,p);
      if (img->val[p] >= Th)
	label->val[p]=1;	      
    }
  }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
    u.x = p%img->ncols;
    u.y = p/img->ncols;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (cost->val[q] > cost->val[p]){
	  tmp = MAX(cost->val[p],abs(img->val[q]-img->val[root->val[p]]));
	  if (tmp < cost->val[q]){
	    if (cost->val[q]!=INT_MAX) RemoveGQueueElem(Q,q);
	    label->val[q] = label->val[p]; cost->val[q]=tmp;
	    root->val[q] = root->val[p];
	    InsertGQueue(&Q,q);
	  }
	}
      }
    }
  }

  DestroyGQueue(&Q);
  DestroyImage(&cost);
  DestroyImage(&root);
  DestroyCurve(&hist);

  return(label);
}

