
#include "mmorphology.h"


Scene *ErodeBinScnRadial(Scene *bin, Voxel C, float r){
  Scene *ero=NULL;
  Vector v;
  Voxel a,b;
  int p,q,d;

  ero = CopyScene(bin);
  for(p=0; p<bin->n; p++){
    if(bin->data[p]==0) continue;
    
    a.x = VoxelX(bin,p);
    a.y = VoxelY(bin,p);
    a.z = VoxelZ(bin,p);

    v.x = (C.x - a.x)*bin->dx;
    v.y = (C.y - a.y)*bin->dy;
    v.z = (C.z - a.z)*bin->dz;
    VectorNormalize(&v);
    
    for(d=-ROUND(r); d<=ROUND(r); d++){
      b.x = a.x + ROUND((v.x*d)/bin->dx);
      b.y = a.y + ROUND((v.y*d)/bin->dy);
      b.z = a.z + ROUND((v.z*d)/bin->dz);
      if(ValidVoxel(bin, b.x,b.y,b.z)){
	q = VoxelAddress(bin, b.x,b.y,b.z);
	if(bin->data[q]==0){ ero->data[p]=0; break; }
      }
      else{ ero->data[p]=0; break; }
    }
  }
  return ero;
}



/* It assumes that the next operation is a erosion */
/*
Scene *DilateBinScn(Scene *bin, Set **seed, float radius){
  Scene *cost=NULL,*root=NULL,*dil=NULL;
  Queue *Q=NULL;
  int i,p,q,sz,xysize,nbuckets;
  Voxel v,w;
  int tmp=INT_MAX,Dx,Dy,Dz;
  float *sq_x=NULL,*sq_y=NULL,*sq_z=NULL;
  float dist,dmax;
  AdjRel3 *A=NULL;
  AdjVxl  *N=NULL;

  SetSceneFrame(bin, 1, 0);
  if(*seed == NULL){
    A = Spheric(1.0);
    N = AdjVoxels(bin, A);
    for(p=0; p<bin->n; p++){
      if(bin->data[p]==0) continue;
      for(i=1; i<N->n; i++){
	q = p + N->dp[i];
	if(bin->data[q]==0){
	  InsertSet(seed,p);
	  break;
	}
      }
    }
    DestroyAdjRel3(&A);
    DestroyAdjVxl(&N);
  }

  dist = (radius*radius);
  A  = Spheric(1.8);
  N = AdjVoxels(bin, A);

  sq_x = AllocFloatArray(bin->xsize);
  sq_y = AllocFloatArray(bin->ysize);
  sq_z = AllocFloatArray(bin->zsize);
  for(i=0; i<bin->xsize; i++) sq_x[i]=(i*i*bin->dx*bin->dx);
  for(i=0; i<bin->ysize; i++) sq_y[i]=(i*i*bin->dy*bin->dy);
  for(i=0; i<bin->zsize; i++) sq_z[i]=(i*i*bin->dz*bin->dz);

  dil  = CopyScene(bin);
  cost = CreateScene(dil->xsize,dil->ysize,dil->zsize);
  root = CreateScene(dil->xsize,dil->ysize,dil->zsize);
  xysize = dil->xsize*dil->ysize;

  sz = FrameSize3(A);
  dmax = MAX(bin->dx, MAX(bin->dy, bin->dz));
  nbuckets  = ROUND(2*sz*dmax*(dil->xsize*bin->dx+
			       dil->ysize*bin->dy+
			       dil->zsize*bin->dz)+
		    3*sz*sz*dmax*dmax)+1;
  Q  = CreateQueue(nbuckets,dil->n);

  for(p=0; p <dil->n; p++){
    if(bin->data[p]==0) cost->data[p]=INT_MAX;
    else                cost->data[p]=0;
  }
  SetSceneFrame(cost, 1, 0);
  
  while (*seed != NULL){
    p=RemoveSet(seed);
    if(bin->data[p]==0) continue;
    cost->data[p]=0;
    root->data[p]=p;
    InsertQueue(Q,cost->data[p]%Q->C.nbuckets,p);
  }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if(cost->data[p] <= dist){

      dil->data[p] = 1;
      w.x = (root->data[p] % xysize) % cost->xsize;
      w.y = (root->data[p] % xysize) / cost->xsize;
      w.z = root->data[p] / xysize;

      for(i=1; i<N->n; i++){
	q = p + N->dp[i];
	if(cost->data[p] < cost->data[q]){

	  v.x = (q % xysize) % cost->xsize;
	  v.y = (q % xysize) / cost->xsize;
	  v.z = q / xysize;
	  Dx  = abs(v.x-w.x);
	  Dy  = abs(v.y-w.y);
	  Dz  = abs(v.z-w.z);
	  tmp = ROUND(sq_x[Dx] + sq_y[Dy] + sq_z[Dz]);
	  if(tmp < cost->data[q]){
	    if(cost->data[q] == INT_MAX)
	      InsertQueue(Q,tmp%Q->C.nbuckets,q);
	    else
	      UpdateQueue(Q,q,cost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	    cost->data[q] = tmp;
	    root->data[q] = root->data[p];
	  }
	}
      }
    } else {
      InsertSet(seed,p);
    }
  }

  free(sq_x);
  free(sq_y);
  free(sq_z);
  DestroyQueue(&Q);
  DestroyScene(&root);
  DestroyScene(&cost);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&N);

  return(dil);
}
*/


/* It assumes that the next operation is a erosion */

Scene *DilateBinScn(Scene *bin, Set **seed, float radius){
  Scene *cost=NULL,*root=NULL,*dil=NULL;
  Queue *Q=NULL;
  int i,p,q,sz,xysize,nbuckets;
  Voxel u,v,w;
  int tmp=INT_MAX,Dx,Dy,Dz;
  float *sq_x=NULL,*sq_y=NULL,*sq_z=NULL;
  float dist,dmax;
  AdjRel3 *A=NULL;

  if(*seed == NULL) {
    A = Spheric(1.0);
    for (u.z=0; u.z < bin->zsize; u.z++) {
      for (u.y=0; u.y < bin->ysize; u.y++){
	for (u.x=0; u.x < bin->xsize; u.x++){
	  p = u.x + bin->tby[u.y]+bin->tbz[u.z];
	  if(bin->data[p]>0){
	    for(i=1; i < A->n; i++){
	      v.x = u.x + A->dx[i];
	      v.y = u.y + A->dy[i];
	      v.z = u.z + A->dz[i];
	      if(ValidVoxel(bin,v.x,v.y,v.z)){
		q = v.x + bin->tby[v.y] + bin->tbz[v.z];
		if(bin->data[q]==0){
		  InsertSet(seed,p);
		  break;
		}
	      }
	    }
	  }
	}
      }
    }
    DestroyAdjRel3(&A);
  }

  dist = (radius*radius);
  A  = Spheric(1.8);

  sq_x = AllocFloatArray(bin->xsize);
  sq_y = AllocFloatArray(bin->ysize);
  sq_z = AllocFloatArray(bin->zsize);
  for(i=0; i<bin->xsize; i++) sq_x[i]=(i*i*bin->dx*bin->dx);
  for(i=0; i<bin->ysize; i++) sq_y[i]=(i*i*bin->dy*bin->dy);
  for(i=0; i<bin->zsize; i++) sq_z[i]=(i*i*bin->dz*bin->dz);

  dil  = CopyScene(bin);
  cost = CreateScene(dil->xsize,dil->ysize,dil->zsize);
  root = CreateScene(dil->xsize,dil->ysize,dil->zsize);
  xysize = dil->xsize*dil->ysize;

  sz = FrameSize3(A);
  dmax = MAX(bin->dx, MAX(bin->dy, bin->dz));
  nbuckets  = ROUND(2*sz*dmax*(dil->xsize*bin->dx+
			       dil->ysize*bin->dy+
			       dil->zsize*bin->dz)+
		    3*sz*sz*dmax*dmax)+1;
  Q = CreateQueue(nbuckets,dil->n);

  for(p=0; p <dil->n; p++){
    if(bin->data[p]==0) cost->data[p]=INT_MAX;
    else                cost->data[p]=0;
  }
  
  while (*seed != NULL){
    p=RemoveSet(seed);
    cost->data[p]=0;
    root->data[p]=p;
    InsertQueue(Q,cost->data[p]%Q->C.nbuckets,p);
  }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if(cost->data[p] <= dist){

      dil->data[p] = 1;

      u.x = (p % xysize) % cost->xsize;
      u.y = (p % xysize) / cost->xsize;
      u.z = p / xysize;

      w.x = (root->data[p] % xysize) % cost->xsize;
      w.y = (root->data[p] % xysize) / cost->xsize;
      w.z = root->data[p] / xysize;

      for(i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	v.z = u.z + A->dz[i];
	if(ValidVoxel(dil,v.x,v.y,v.z)){
	  q = v.x + dil->tby[v.y] + dil->tbz[v.z];

	  if(cost->data[p] < cost->data[q]){
	    Dx  = abs(v.x-w.x);
	    Dy  = abs(v.y-w.y);
	    Dz  = abs(v.z-w.z);
	    tmp = ROUND(sq_x[Dx] + sq_y[Dy] + sq_z[Dz]);
	    if(tmp < cost->data[q]){
	      if(cost->data[q] == INT_MAX)
		InsertQueue(Q,tmp%Q->C.nbuckets,q);
	      else
		UpdateQueue(Q,q,cost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	      cost->data[q] = tmp;
	      root->data[q] = root->data[p];
	    }
	  }
	}
      }
    } else {
      InsertSet(seed,p);
    }
  }

  free(sq_x);
  free(sq_y);
  free(sq_z);
  DestroyQueue(&Q);
  DestroyScene(&root);
  DestroyScene(&cost);
  DestroyAdjRel3(&A);

  return(dil);
}



/* It assumes that the next operation is a dilation */

Scene *ErodeBinScn(Scene *bin, Set **seed, float radius){
  Scene *cost=NULL,*root=NULL,*ero=NULL;
  Queue *Q=NULL;
  int i,p,q,sz,xysize,nbuckets;
  Voxel u,v,w;
  int tmp=INT_MAX,Dx,Dy,Dz;
  float *sq_x=NULL,*sq_y=NULL,*sq_z=NULL;
  float dist,dmax;
  AdjRel3 *A=NULL;

  if (*seed == NULL) {
    A      = Spheric(1.0);
    for (u.z=0; u.z < bin->zsize; u.z++) {
      for (u.y=0; u.y < bin->ysize; u.y++){
	for (u.x=0; u.x < bin->xsize; u.x++){
	  p = u.x + bin->tby[u.y]+bin->tbz[u.z];
	  if (bin->data[p]==0){
	    for (i=1; i < A->n; i++){
	      v.x = u.x + A->dx[i];
	      v.y = u.y + A->dy[i];
	      v.z = u.z + A->dz[i];
	      if (ValidVoxel(bin,v.x,v.y,v.z)){
		q = v.x + bin->tby[v.y] + bin->tbz[v.z];
		if (bin->data[q]==1){
		  InsertSet(seed,p);
		  break;
		}
	      }
	    }
	  }
	}
      }
    }
    DestroyAdjRel3(&A);
  }

  dist = (radius*radius);
  A  = Spheric(1.8);

  sq_x = AllocFloatArray(bin->xsize);
  sq_y = AllocFloatArray(bin->ysize);
  sq_z = AllocFloatArray(bin->zsize);
  for(i=0; i<bin->xsize; i++) sq_x[i]=(i*i*bin->dx*bin->dx);
  for(i=0; i<bin->ysize; i++) sq_y[i]=(i*i*bin->dy*bin->dy);
  for(i=0; i<bin->zsize; i++) sq_z[i]=(i*i*bin->dz*bin->dz);

  ero  = CopyScene(bin);
  cost = CreateScene(ero->xsize,ero->ysize,ero->zsize);
  root = CreateScene(ero->xsize,ero->ysize,ero->zsize);
  xysize = ero->xsize*ero->ysize;

  sz = FrameSize3(A);
  dmax = MAX(bin->dx, MAX(bin->dy, bin->dz));
  nbuckets  = ROUND(2*sz*dmax*(ero->xsize*bin->dx+
			       ero->ysize*bin->dy+
			       ero->zsize*bin->dz)+
		    3*sz*sz*dmax*dmax)+1;
  Q  = CreateQueue(nbuckets,ero->n);
  
  for(p=0; p<cost->n; p++){
    if(bin->data[p]==1) cost->data[p]=INT_MAX;
    else                cost->data[p]=0;
  }

  while (*seed != NULL){
    p=RemoveSet(seed);
    cost->data[p]=0;
    root->data[p]=p;
    InsertQueue(Q,cost->data[p]%Q->C.nbuckets,p);
  }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (cost->data[p] <= dist){

      ero->data[p] = 0;

      u.x = (p % xysize) % cost->xsize;
      u.y = (p % xysize) / cost->xsize;
      u.z = p / xysize;

      w.x = (root->data[p] % xysize) % cost->xsize;
      w.y = (root->data[p] % xysize) / cost->xsize;
      w.z = root->data[p] / xysize;

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	v.z = u.z + A->dz[i];
	if (ValidVoxel(ero,v.x,v.y,v.z)){
	  q = v.x + ero->tby[v.y] + ero->tbz[v.z];
	  if(cost->data[p] < cost->data[q]){
	    Dx  = abs(v.x-w.x);
	    Dy  = abs(v.y-w.y);
	    Dz  = abs(v.z-w.z);
	    tmp = ROUND(sq_x[Dx] + sq_y[Dy] + sq_z[Dz]);
	    if (tmp < cost->data[q]){
	      if (cost->data[q] == INT_MAX)
		InsertQueue(Q,tmp%Q->C.nbuckets,q);
	      else
		UpdateQueue(Q,q,cost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	      cost->data[q]  = tmp;
	      root->data[q]  = root->data[p];
	    }
	  }
	}
      }
    } else {
      InsertSet(seed,p);
    }
  }

  free(sq_x);
  free(sq_y);
  free(sq_z);
  DestroyQueue(&Q);
  DestroyScene(&root);
  DestroyScene(&cost);
  DestroyAdjRel3(&A);

  return(ero);
}


Scene *CloseBinScn(Scene *bin, float radius){
  Scene *close=NULL,*dil=NULL;
  Set *seed=NULL;

  dil   = DilateBinScn(bin,&seed,radius);
  close = ErodeBinScn(dil,&seed,radius);
  DestroyScene(&dil);
  DestroySet(&seed);

  return(close);
}


Scene *OpenBinScn(Scene *bin, float radius){
  Scene *open=NULL,*ero=NULL;
  Set *seed=NULL;

  ero   = ErodeBinScn(bin,&seed,radius);
  open  = DilateBinScn(ero,&seed,radius);
  DestroyScene(&ero);
  DestroySet(&seed);

  return(open);
}

