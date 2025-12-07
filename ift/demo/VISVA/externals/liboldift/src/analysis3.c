#include "analysis3.h"
#include "mathematics3.h"
#include "segmentation3.h"
#include "scene.h"
#include "queue.h"

void iftDistTrans3(AnnScn *ascn, AdjRel3 *A, char side)
{
  AdjRel3 *A6=Spheric(1.0);
  Voxel u,v;
  int s,t,i,n,p;
  Scene *scn=ascn->scn;
  int qsize,fsize;

  n = ascn->scn->xsize*ascn->scn->ysize*ascn->scn->zsize;

  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++)
      if (ascn->scn->data[p] == 0)
	ascn->cost->data[p] = 0;
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++)
      if (ascn->scn->data[p] != 0)
	ascn->cost->data[p] = 0;
    break;
  }
  
  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++){
	s = u.x + scn->tby[u.y] + scn->tbz[u.z];
	if (scn->data[s] != 0){ 
	  for (i=1; i < A6->n; i++){
	    v.x = u.x + A6->dx[i];
	    v.y = u.y + A6->dy[i];
	    v.z = u.z + A6->dz[i];
	    if (ValidVoxel(scn,v.x,v.y,v.z)){
	      t = v.x + scn->tby[v.y] + scn->tbz[v.z];
	      if (scn->data[t]==0){
		AddSeed3(ascn,u.x,u.y,u.z,0,s,s);

		break;
	      }
	    } else {
	      AddSeed3(ascn,u.x,u.y,u.z,0,s,s);

	      break;
	    }
	  }
	}
      }
  DestroyAdjRel3(&A6);
  fsize = FrameSize3(A);
  qsize = 2*fsize*(scn->xsize+scn->ysize+scn->zsize)+3*fsize*fsize;
  IFT3(ascn,A,FEucl3,qsize);
}

void iftGenDistTrans3(AnnScn *ascn, AdjRel3 *A, PathCost3 Pcost)
{
  AdjRel3 *A6=Spheric(1.0);
  Voxel u,v;
  int s,t,i;
  Scene *scn=ascn->scn;
  int qsize,fsize;

  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++){
	s = u.x + scn->tby[u.y] + scn->tbz[u.z];
	if (scn->data[s] != 0){ 
	  for (i=1; i < A6->n; i++){
	    v.x = u.x + A6->dx[i];
	    v.y = u.y + A6->dy[i];
	    v.z = u.z + A6->dz[i];
	    if (ValidVoxel(scn,v.x,v.y,v.z)){
	      t = v.x + scn->tby[v.y] + scn->tbz[v.z];
	      if (scn->data[t]==0){
		AddSeed3(ascn,u.x,u.y,u.z,0,s,s);
		break;
	      }
	    } else {
	      AddSeed3(ascn,u.x,u.y,u.z,0,s,s);
	      break;
	    }
	  }
	}
      }
  DestroyAdjRel3(&A6);
  fsize = FrameSize3(A);
  qsize = 2*fsize*(scn->xsize+scn->ysize+scn->zsize)+3*fsize*fsize;
  IFT3(ascn,A,Pcost,qsize);
}

Scene *DistTrans3(Scene *bin, AdjRel3 *A, char side, char sign)
{
  Scene *Dx=NULL,*Dy=NULL,*Dz=NULL,*fbin,*fcont,*fcost,*cost;
  Queue *Q=NULL;
  int i,p,q,n,sz;
  int *sq=NULL,tmp=INT_MAX,dx,dy,dz;
  AdjVxl *N;
  AdjRel3 *A6 = Spheric(1.0);

  n  = MAX(bin->xsize,MAX(bin->ysize,bin->zsize));
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  sz = FrameSize3(A);  
  fbin = AddFrame3(bin,sz,0);
  fcont = GetBorder3(fbin,A6);
  fcost = AddFrame3(bin,sz,INT_MIN);
  Dx = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);
  Dy = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);  
  Dz = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);  
  N  = AdjVoxels(fcost,A);
  n  = fcost->xsize*fcost->ysize*fcost->zsize;
  Q = CreateQueue(2*sz*(bin->xsize+bin->ysize+bin->zsize)+3*sz*sz,n);

  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->data[p] != 0){
	if (fcont->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	} else
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = 0;
      }
    }
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->data[p] == 0){
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fcont->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	}else
	  fcost->data[p] = 0;
      }
    }
    break;
  case BOTH:
  default:    
    for(p = 0; p < n; p++){
      if (fcont->data[p] > 0){
	fcost->data[p]=0;    
	InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
      }else{ 
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p]=INT_MAX;    
      }
    }
  }
  DestroyScene(&fcont);
  DestroyScene(&fbin);

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->data[p] < fcost->data[q]){
	dx  = Dx->data[p] + abs(A->dx[i]);
	dy  = Dy->data[p] + abs(A->dy[i]);
	dz  = Dz->data[p] + abs(A->dz[i]);
	tmp = sq[dx] + sq[dy] + sq[dz];
	if (tmp < fcost->data[q]){
	  if (fcost->data[q] == INT_MAX)
	    InsertQueue(Q,tmp%Q->C.nbuckets,q);
	  else 
	    UpdateQueue(Q,q,fcost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	  fcost->data[q]  = tmp;	    
	  Dx->data[q] = dx;
	  Dy->data[q] = dy;
	  Dz->data[q] = dz;	  
	}
      }
    }
  }
  
  DestroyQueue(&Q);
  DestroyAdjVxl(&N);
  DestroyAdjRel3(&A6);
  cost = RemFrame3(fcost,sz);
  free(sq);
  DestroyScene(&Dx);
  DestroyScene(&Dy);
  DestroyScene(&Dz);
  DestroyScene(&fcost);
  // sign scene
  if (sign != 0){
    n  = cost->xsize*cost->ysize*cost->zsize;  
    if (side != INTERIOR)
      for (i=0; i<n; i++) {	
	if (bin->data[i] == 0) {
	  cost->data[i] = -cost->data[i];
	}
      }
  }

  return(cost);
}

Scene *TDistTrans3(Scene *bin, AdjRel3 *A, char side, int limit, char sign)
{
  Scene *Dx=NULL,*Dy=NULL,*Dz=NULL,*fbin,*fcont,*fcost,*cost;
  Queue *Q=NULL;
  int i,p,q,n,sz;
  int *sq=NULL,tmp=INT_MAX,dx,dy,dz;
  AdjVxl *N;
  AdjRel3 *A6 = Spheric(1.0);

  n  = MAX(bin->xsize,MAX(bin->ysize,bin->zsize));
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  sz = FrameSize3(A);  
  fbin = AddFrame3(bin,sz,0);
  fcont = GetBorder3(fbin,A6);
  fcost = AddFrame3(bin,sz,INT_MIN);
  Dx = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);
  Dy = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);  
  Dz = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);  
  N  = AdjVoxels(fcost,A);
  n  = fcost->xsize*fcost->ysize*fcost->zsize;
  Q = CreateQueue(2*sz*(bin->xsize+bin->ysize+bin->zsize)+3*sz*sz,n);

  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->data[p] != 0){
	if (fcont->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	} else
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = 0;
      }
    }
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++){
      if (fbin->data[p] == 0){
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fcont->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	}else
	  fcost->data[p] = 0;
      }
    }
    break;
  case BOTH:
  default:    
    for(p = 0; p < n; p++){
      if (fcont->data[p] > 0){
	fcost->data[p]=0;    
	InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
      }else{ 
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p]=INT_MAX;    
      }
    }
  }
  DestroyScene(&fcont);
  DestroyScene(&fbin);

  
  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (fcost->data[p] < fcost->data[q]){
	dx  = Dx->data[p] + abs(A->dx[i]);
	dy  = Dy->data[p] + abs(A->dy[i]);
	dz  = Dz->data[p] + abs(A->dz[i]);
	tmp = sq[dx] + sq[dy] + sq[dz];
	if (tmp < fcost->data[q] && tmp <= limit){
	  if (fcost->data[q] == INT_MAX)
	    InsertQueue(Q,tmp%Q->C.nbuckets,q);
	  else 
	    UpdateQueue(Q,q,fcost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	  fcost->data[q]  = tmp;	    
	  Dx->data[q] = dx;
	  Dy->data[q] = dy;
	  Dz->data[q] = dz;	  
	}
      }
    }
  }
  DestroyQueue(&Q);
  DestroyAdjVxl(&N);
  DestroyAdjRel3(&A6);
  cost = RemFrame3(fcost,sz);
  free(sq);
  DestroyScene(&Dx);
  DestroyScene(&Dy);
  DestroyScene(&Dz);
  DestroyScene(&fcost);

  // Eliminate infinite values */

  for (i=0; i<n; i++) {
    if (cost->data[i]==INT_MAX)
      cost->data[i]=0;
  }

  // sign scene
  if (sign != 0){
    n  = cost->xsize*cost->ysize*cost->zsize;  
    if (side != INTERIOR)
      for (i=0; i<n; i++) {
	if (bin->data[i] == 0) {
	  cost->data[i] = -cost->data[i];
	}
      }
  }
  return(cost);
}

Scene *Label2Border(Scene *lscn, AdjRel3 *A, char side, int limit)
{
  Scene *Dx=NULL,*Dy=NULL,*Dz=NULL,*flscn,*fcost;
  Scene *border, *fborder;
  Queue *Q=NULL;
  int i,p,q,n,sz;
  int *sq=NULL,tmp=INT_MAX,dx,dy,dz;
  AdjVxl *N;
  AdjRel3 *A6 = Spheric(1.0);

  n  = MAX(lscn->xsize,MAX(lscn->ysize,lscn->zsize));
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  sz = FrameSize3(A);  
  flscn = AddFrame3(lscn,sz,0);
  fborder = GetBorder3(flscn,A6);
  fcost = AddFrame3(lscn,sz,INT_MIN);
  Dx = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);
  Dy = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);  
  Dz = CreateScene(fcost->xsize,fcost->ysize,fcost->zsize);  
  N  = AdjVoxels(fcost,A);
  n  = fcost->xsize*fcost->ysize*fcost->zsize;
  Q = CreateQueue(2*sz*(lscn->xsize+lscn->ysize+lscn->zsize)+3*sz*sz,n);

  switch (side) {
  case INTERIOR:
    for(p = 0; p < n; p++){
      if (flscn->data[p] != 0){
	if (fborder->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	} else
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = 0;
      }
    }
    break;
  case EXTERIOR:
    for(p = 0; p < n; p++){
      if (flscn->data[p] == 0){
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p] = INT_MAX;	  
      }else{
	if (fborder->data[p]>0){
	  fcost->data[p]=0;    
	  InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
	}else
	  fcost->data[p] = 0;
      }
    }
    break;
  case BOTH:
  default:    
    for(p = 0; p < n; p++){
      if (fborder->data[p] > 0){
	fcost->data[p]=0;    
	InsertQueue(Q,fcost->data[p]%Q->C.nbuckets,p);
      }else{ 
	if (fcost->data[p]!=INT_MIN)
	  fcost->data[p]=INT_MAX;    
      }
    }
  }
  
  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    for (i=1; i < N->n; i++){
      q = p + N->dp[i];
      if (flscn->data[q] == flscn->data[p]) {
        if (fcost->data[p] < fcost->data[q]) {	  
  	  dx  = Dx->data[p] + abs(A->dx[i]);
	  dy  = Dy->data[p] + abs(A->dy[i]);
	  dz  = Dz->data[p] + abs(A->dz[i]);
	  tmp = sq[dx] + sq[dy] + sq[dz];
	  if (tmp < fcost->data[q] && tmp <= limit){
	    if (fcost->data[q] == INT_MAX)
	      InsertQueue(Q,tmp%Q->C.nbuckets,q);
	    else 
	      UpdateQueue(Q,q,fcost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	    fcost->data[q]  = tmp;	    
	    Dx->data[q] = dx;
	    Dy->data[q] = dy;
	    Dz->data[q] = dz;
            fborder->data[q] = fborder->data[p];
	  } 
	}
      }
    }
  }
  DestroyQueue(&Q);
  DestroyAdjVxl(&N);
  border = RemFrame3(fborder,sz);
  free(sq);
  DestroyScene(&Dx);
  DestroyScene(&Dy);
  DestroyScene(&Dz);
  DestroyScene(&fcost);
  DestroyScene(&fborder);
  DestroyScene(&flscn);
  DestroyAdjRel3(&A6);
  return(border);
}

Scene *Area3(Scene *bin)
{
  int p,n;
  Scene *label,*area;
  Curve *hist;
  AdjRel3 *A;

  A     = Spheric(1.8);
  label = LabelBinComp3(bin,A);
  n     = label->xsize*label->ysize*label->zsize;
  area  = CreateScene(label->xsize,label->ysize,label->zsize);
  hist  = Histogram3(label);
  for (p=0; p < n; p++)
    if (label->data[p] > 0)
      area->data[p] = hist->Y[label->data[p]];

  DestroyCurve(&hist);
  DestroyScene(&label);
  DestroyAdjRel3(&A);

  return(area);
}
