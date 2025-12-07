#include "segmentation3.h"
#include "geometry.h"
#include "queue.h"
#include "curve.h"
#include "radiometric3.h"

Set    *TreeRemoval(AnnScn *ascn, AdjRel3 *A)
{
  int r,i,n,aux,xysize;
  int *trees,p,q,first,last;
  Voxel u,v;
  BMap *Rcolor;
  BMap *Fcolor;
  Set *seedsorfrontier=NULL;

  xysize = ascn->scn->xsize*ascn->scn->ysize;
  n      = xysize*ascn->scn->zsize;
  Rcolor = BMapNew(n);// with flag initialization to 0
  Fcolor = BMapNew(n);

  trees      = AllocIntArray(n);
  first      = last = 0;
  while (ascn->mark != NULL){
    p = RemoveSet(&(ascn->mark));
    r = ascn->label->data[p];// assuming label map is root map
    if (ascn->cost->data[r] != INT_MAX){
      BMapSet(Rcolor,r,GRAY);
      trees[last]=r; last++;
      ascn->cost->data[r]=INT_MAX; ascn->pred->data[r]=r;
    }
  }
  while (first != last){ 
    p = trees[first];
    first++;

    aux = p%xysize;
    u.x = aux%ascn->scn->xsize;
    u.y = aux/ascn->scn->xsize;
    u.z = p/xysize;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(ascn->scn,v.x,v.y,v.z)){
	q = v.x + ascn->scn->tby[v.y] + ascn->scn->tbz[v.z];
	if (ascn->pred->data[q]==p){
	  ascn->cost->data[q]=INT_MAX; ascn->pred->data[q]=q;
	  trees[last]=q; last++; 
	}else{
	  if (BMapGet(Rcolor,ascn->label->data[q])==WHITE && BMapGet(Fcolor,q)==WHITE){
	    InsertSet(&seedsorfrontier,q);
	    BMapSet(Fcolor,q,GRAY);
	  }
	}
      }
    }
  }

  while (ascn->seed != NULL){
    p = RemoveSet(&(ascn->seed));
    
    if(ascn->scn->data[p] < ascn->cost->data[p]){
      ascn->cost->data[p] = ascn->scn->data[p];
      ascn->label->data[p] = p;
      ascn->pred->data[p] = p;
      if (BMapGet(Fcolor,p)==WHITE){// to insert each seed only once
	InsertSet(&seedsorfrontier,p);
	BMapSet(Fcolor,p,GRAY);
      }
    }
  }
  
  
  free(trees);
  BMapDestroy(Rcolor);
  BMapDestroy(Fcolor);
  return(seedsorfrontier);
}

void DIFT(AnnScn *ascn, AdjRel3 *A)
{
  Queue *Q=NULL;
  int i,p,q,n,cost,aux,xysize;
  Voxel u,v;
  Set *seedsorfrontier=NULL;
  
  seedsorfrontier = TreeRemoval(ascn,A);

  xysize = ascn->scn->xsize*ascn->scn->ysize;
  n      = xysize*ascn->scn->zsize;
  Q = CreateQueue(MaximumValue3(ascn->scn)+1,n);// initialization of each element with color = WHITE

  while (seedsorfrontier != NULL){
    p = RemoveSet(&seedsorfrontier);
    InsertQueue(Q,ascn->cost->data[p],p);
  }

  while((p=RemoveQueue(Q))!=NIL) {
    aux = p%xysize;
    u.x = aux%ascn->scn->xsize;
    u.y = aux/ascn->scn->xsize;
    u.z = p/xysize;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(ascn->scn,v.x,v.y,v.z)){
	q = v.x + ascn->scn->tby[v.y] + ascn->scn->tbz[v.z];
	if (Q->L.elem[q].color != BLACK){
	  cost = MAX(ascn->cost->data[p],ascn->scn->data[q]);
	  if ((cost < ascn->cost->data[q])||
	      (ascn->pred->data[q] == p)){
	    if (Q->L.elem[q].color == GRAY)
	      RemoveQueueElem(Q,q,ascn->cost->data[q]); // color of q = BLACK
	    ascn->cost->data[q]  = cost;
	    ascn->pred->data[q]  = p;
	    ascn->label->data[q] = ascn->label->data[p];
	    InsertQueue(Q,cost,q); // color of q = GRAY
	  }
	}
      }
    }
  }
  DestroyQueue(&Q);
}


Set *BTreeRemoval(AnnScn *ascn, AdjRel3 *A, Border *border, BMap *bordermap)
{
  int r,i,n,aux,xysize;
  int *trees,p,q,first,last;
  Voxel u,v;
  BMap *color;
  Set *seedsorfrontier=NULL;

  xysize = ascn->scn->xsize*ascn->scn->ysize;
  n      = xysize*ascn->scn->zsize;
  color = BMapNew(n);// with flag initialization to 0
  while (ascn->seed != NULL){
    p = RemoveSet(&(ascn->seed));
    if (BMapGet(color,p)==WHITE){// to insert each seed only once
      InsertSet(&seedsorfrontier,p);
      BMapSet(color,p,GRAY);
    }
  }
  trees      = AllocIntArray(n);
  first      = last = 0;
  while (ascn->mark != NULL){
    p = RemoveSet(&(ascn->mark));
    r = ascn->label->data[p];// assuming label map is root map
    if (ascn->cost->data[r] != INT_MAX){
      BMapSet(color,r,GRAY);
      trees[last]=r; last++;
      ascn->cost->data[r]=INT_MAX; ascn->pred->data[r]=r;
    }
  }
  while (first != last){ 
    p = trees[first];
    first++;

		if(BMapGet(bordermap,p)) { // p is a border voxel
			RemoveBorder(border,p);
			BMapSet(bordermap,p,0);
		}

    aux = p%xysize;
    u.x = aux%ascn->scn->xsize;
    u.y = aux/ascn->scn->xsize;
    u.z = p/xysize;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(ascn->scn,v.x,v.y,v.z)){
				q = v.x + ascn->scn->tby[v.y] + ascn->scn->tbz[v.z];
				if (ascn->pred->data[q]==p){
					ascn->cost->data[q]=INT_MAX; ascn->pred->data[q]=q;
					trees[last]=q; last++; 
				}else{
				  if ((BMapGet(color,q)==WHITE)&&
				      (BMapGet(color,ascn->label->data[q])==WHITE)){
				    InsertSet(&seedsorfrontier,q);
				    BMapSet(color,q,GRAY);
				    if(BMapGet(bordermap,q)) { // q is a border voxel
				      RemoveBorder(border,q);
				      BMapSet(bordermap,q,0);
				    }
				  }
				}
      }
    }
  }
  free(trees);
  BMapDestroy(color);
  return(seedsorfrontier);
}

void BDIFT(AnnScn *ascn, AdjRel3 *A, BMap *isobj, BMap *bordermap, Border *border)
{
  Queue *Q=NULL;
  int i,p,q,n,cost,aux,xysize,lbl;
  Voxel u,v;
  Set *seedsorfrontier=NULL;
  
  seedsorfrontier = BTreeRemoval(ascn,A,border,bordermap);

  xysize = ascn->scn->xsize*ascn->scn->ysize;
  n      = xysize*ascn->scn->zsize;
  Q = CreateQueue(MaximumValue3(ascn->scn)+1,n);// initialization of each element with color = WHITE

  while (seedsorfrontier != NULL){
    p = RemoveSet(&seedsorfrontier);
    InsertQueue(Q,ascn->cost->data[p],p);
  }

  while((p=RemoveQueue(Q))!=NIL) {
    aux = p%xysize;
    u.x = aux%ascn->scn->xsize;
    u.y = aux/ascn->scn->xsize;
    u.z = p/xysize;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(ascn->scn,v.x,v.y,v.z)){
	q = v.x + ascn->scn->tby[v.y] + ascn->scn->tbz[v.z];
	
	if ((ascn->cost->data[p] < ascn->cost->data[q])||
	    (ascn->pred->data[q] == p)){
	  cost = MAX(ascn->cost->data[p],ascn->scn->data[q]);
	  if ((cost < ascn->cost->data[q])||
	      (ascn->pred->data[q] == p)){
	    if (Q->L.elem[q].color == GRAY)
	      RemoveQueueElem(Q,q,ascn->cost->data[q]); // color of q = BLACK
	    ascn->cost->data[q]  = cost;
	    ascn->pred->data[q]  = p;
	    ascn->label->data[q] = ascn->label->data[p];						
	    lbl = BMapGet(isobj,p);
	    BMapSet(isobj,q,lbl); // propagate "isobject" label (0 or 1)
	    InsertQueue(Q,cost,q); // color of q = GRAY
	  }
	}
	
	else {
	  if((BMapGet(isobj,p))&&(!BMapGet(isobj,q))) { // p is object and q is background
	    if (BMapGet(bordermap,p)==0){// to insert each border voxel only once
	      InsertBorder(border,p);
	      BMapSet(bordermap,p,1);
	    }
	  }
	  else if((BMapGet(isobj,q))&&(!BMapGet(isobj,p))) { // q is object and p is background
	    if (BMapGet(bordermap,q)==0){// to insert each border voxel only once
	      InsertBorder(border,q);
	      BMapSet(bordermap,q,1);
	    }
	  }
	}
      }
    }
  }
  DestroyQueue(&Q);
}

Scene *Threshold3(Scene *scn, int lower, int higher)
{
  Scene *bin=NULL;
  int p,n;

  bin = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  bin->dx = scn->dx;
  bin->dy = scn->dy;
  bin->dz = scn->dz;
  n = scn->xsize*scn->ysize*scn->zsize;
  for (p=0; p < n; p++)
    if ((scn->data[p] >= lower)&&(scn->data[p] <= higher))
      bin->data[p]=1;
  bin->maxval = 1;
  return(bin);
}

int AutoThreshold3(Scene *scn)
{
  Curve *hist=NULL;
  int i,n,xmin = 0;
  long dd, d, d0, dmin;

  hist = Histogram3(scn);

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

Scene *ThresholdMask3(Scene *scn, Scene *mask, int lower, int higher)
{
  Scene *bin=NULL;
  int p,n,max=INT_MIN;

  bin = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  bin->dx = scn->dx;
  bin->dy = scn->dy;
  bin->dz = scn->dz;
  n = scn->xsize*scn->ysize*scn->zsize;
  for (p=0; p < n; p++)
    if ((scn->data[p] >= lower)&&(scn->data[p] <= higher)&&mask->data[p]) {
      bin->data[p]=mask->data[p];
      max = MAX(max,mask->data[p]);
    }
  bin->maxval = max;
  return(bin);
}

Scene *Highlight3(Scene *scn, Scene *label, int value)
{
  Scene *hscn=NULL;
  int p,q,i;
  AdjRel3 *A=NULL;
  Voxel u,v;

  hscn = CopyScene(scn);
  A    = Spheric(1.0);
  for (u.z=0; u.z < hscn->zsize; u.z++){
    for (u.y=0; u.y < hscn->ysize; u.y++){
      for (u.x=0; u.x < hscn->xsize; u.x++){
	p = u.x + hscn->tby[u.y]+hscn->tbz[u.z];
	for (i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(hscn,v.x,v.y,v.z)){
	    q = v.x + hscn->tby[v.y]+hscn->tbz[v.z];
	    if (label->data[p] > label->data[q]){
	      hscn->data[p] = value;
	    break;
	    }
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);

  return(hscn);
}

Scene *LabelComp3(Scene *scn, AdjRel3 *A, int thres)
{
  Scene *label=NULL;
  int i,j,n,p,q,l=1,xysize;
  int *FIFO=NULL;
  int first=0,last=0;
  Voxel u,v;  

  label  = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  xysize = scn->xsize*scn->ysize;
  n      = xysize*scn->zsize;
  FIFO   = AllocIntArray(n);

  for (j=0; j < n; j++){
    if ((label->data[j]==0)){
      label->data[j]=l;
      FIFO[last]=j;      
      last++;      
      while(first != last){
	p = FIFO[first];
	first++;
	u.x = (p%xysize)%scn->xsize;
	u.y = (p%xysize)/scn->xsize;
	u.z = p/xysize;
	for (i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(scn,v.x,v.y,v.z)){
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    if ((fabs(scn->data[q]-scn->data[p])<=thres) && (label->data[q] == 0)){
	      label->data[q] = label->data[p];
	      FIFO[last] = q;
	      last++;
	    }
	  }
	}
      }
      l++;
      first=last=0;
    }
  }

  free(FIFO);

  return(label);
}

Scene *LabelBinComp3(Scene *bin, AdjRel3 *A)
{
  Scene *label=NULL;
  int i,j,n,p,q,l=1,xysize;
  int *FIFO=NULL;
  int first=0,last=0;
  Voxel u,v;  

  label  = CreateScene(bin->xsize,bin->ysize,bin->zsize);
  xysize = bin->xsize*bin->ysize;
  n      = xysize*bin->zsize;
  FIFO   = AllocIntArray(n);

  for (j=0; j < n; j++){
    if ((bin->data[j]==1)&&(label->data[j]==0)){
      label->data[j]=l;
      FIFO[last]=j;      
      last++;      
      while(first != last){
	p = FIFO[first];
	first++;
	u.x = (p%xysize)%bin->xsize;
	u.y = (p%xysize)/bin->xsize;
	u.z = p/xysize;
	for (i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(bin,v.x,v.y,v.z)){
	    q = v.x + bin->tby[v.y] + bin->tbz[v.z];
	    if ((bin->data[q]==1)&&(label->data[q] == 0)){
	      label->data[q] = label->data[p];
	      FIFO[last] = q;
	      last++;
	    }
	  }
	}
      }
      l++;
      first=last=0;
    }
  }

  free(FIFO);

  return(label);
}

Scene *WaterGray3(Scene *scn, Scene *marker, AdjRel3 *A)
{
  Scene  *cost=NULL,*label=NULL;
  GQueue *Q=NULL;
  int i,p,q,tmp,n,r=1,sz;
  Voxel u,v;
  
  n     = scn->xsize*scn->ysize*scn->zsize;
  sz    = (scn->xsize * scn->ysize);
  cost  = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  label = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  Q     = CreateGQueue(MaximumValue3(marker)+2,n,cost->data);
  for (p=0; p < n; p++) {
    cost->data[p]=marker->data[p]+1;
    InsertGQueue(&Q,p);
  }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
    if (label->data[p]==0) {
      cost->data[p]=scn->data[p];
      label->data[p]=r;
      r++;
    }
    u.z =  p/sz;
    u.x = (p%sz)%scn->xsize;
    u.y = (p%sz)/scn->xsize;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){	
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (cost->data[q] > cost->data[p]){
	  tmp = MAX(cost->data[p],scn->data[q]);
	  if (tmp < cost->data[q]){
	    UpdateGQueue(&Q,q,tmp);
	    label->data[q] = label->data[p];
	  }
	}
      }
    }
  }
  
  DestroyGQueue(&Q);
  DestroyScene(&cost);

  return(label);
}

Scene *iftThres3(Scene *scn, AdjRel3 *A)
{
  int     Th=0, Tl=0, Imax = MaximumValue3(scn);
  Scene  *cost=NULL,*label=NULL,*root=NULL;
  GQueue *Q=NULL;
  int     p,q,i,tmp,n,sz;
  Voxel   u,v;
  Curve   *hist=NormAccHistogram3(scn);

  for (i=hist->n-1; i > 0; i--){
    if (hist->Y[i] < 0.95){
      Th = i;
      break;
    }
  }
  for (i=0; i > hist->n; i++){
    if (hist->Y[i] > 0.05){
      Tl = i;
      break;
    }
  }
  printf("Tl %d Th %d\n",Tl,Th);
 
  n     = scn->xsize*scn->ysize*scn->zsize;
  sz    = (scn->xsize * scn->ysize);
  cost  = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  label = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  root  = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  Q     = CreateGQueue(2*Imax+1,n,cost->data);

  // Initialization

  for (p=0; p < n; p++) {
    label->data[p]=0; root->data[p]=p;	
    if ((scn->data[p] > Tl)&&
	(scn->data[p] < Th)){
      cost->data[p]=INT_MAX; 
    }else{
      cost->data[p]=0; 
      InsertGQueue(&Q,p);
      if (scn->data[p] >= Th)
	label->data[p]=1;	      
    }
  }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
    tmp = p%sz;
    u.x = tmp%scn->xsize;
    u.y = tmp/scn->xsize;
    u.z = p/sz;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (cost->data[q] > cost->data[p]){
	  tmp = MAX(cost->data[p],abs(scn->data[q]-scn->data[root->data[p]])+abs(scn->data[q]-scn->data[p]));
	  if (tmp < cost->data[q]){
	    if (cost->data[q]!=INT_MAX) RemoveGQueueElem(Q,q);
	    label->data[q] = label->data[p]; cost->data[q]=tmp;
	    root->data[q] = root->data[p];
	    InsertGQueue(&Q,q);
	  }
	}
      }
    }
  }

  DestroyGQueue(&Q);
  DestroyScene(&cost);
  DestroyScene(&root);
  DestroyCurve(&hist);

  return(label);
}



int Otsu3(Scene *scn)
{
  Curve *hist=NormHistogram3(scn);
  double p1,p2,m1,m2,s1,s2,J,Jmax=-1.0;
  int i,T,Topt=0,Imax=MaximumValue3(scn);
  
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
