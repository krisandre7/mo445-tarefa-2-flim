
#include "adjacency_addons.h"


AdjRel3 *SceneSphericalAdjRel3(Scene *scn, float r){
  AdjRel3 *A=NULL;
  float rx,ry,rz;
  rx = MAX(r/scn->dx, 1.0);
  ry = MAX(r/scn->dy, 1.0);
  rz = MAX(r/scn->dz, 1.0);
  if(scn->dx==scn->dy && scn->dx==scn->dz)
    A = Spheric(rx);
  else
    A = Ellipsoid(rx, ry, rz);
  return A;
}


AdjRel3 *SceneSphericalGridAdjRel3(Scene *scn, float r,
				   float spacement){
  AdjRel3 *A=NULL;
  int ri,i;

  ri = ROUND(r/spacement);
  A = Spheric((float)ri);

  for(i=1; i<A->n; i++){
    A->dx[i] = ROUND((float)A->dx[i]*spacement/scn->dx);
    A->dy[i] = ROUND((float)A->dy[i]*spacement/scn->dy);
    A->dz[i] = ROUND((float)A->dz[i]*spacement/scn->dz);
  }
  return A;
}


void     ScaleAdjRel3(AdjRel3 *A,
		      float Sx, float Sy, float Sz){
  int i;
  for(i=0; i<A->n; i++){
    A->dx[i] = ROUND((float)A->dx[i]*Sx);
    A->dy[i] = ROUND((float)A->dy[i]*Sy);
    A->dz[i] = ROUND((float)A->dz[i]*Sz);
  }
}


void     XClipAdjRel3(AdjRel3 *A, int lower, int higher){
  AdjRel3 *B=NULL;
  int i,j,n=0;

  B = CloneAdjRel3(A);
  for(i=0; i<B->n; i++){
    if(B->dx[i]>=lower && B->dx[i]<=higher) n++;
  }
  A->n  = n;
  A->dx = (int *)realloc(A->dx, n*sizeof(int));
  A->dy = (int *)realloc(A->dy, n*sizeof(int));
  A->dz = (int *)realloc(A->dz, n*sizeof(int));
  if(A->dx == NULL) Error(MSG1,"XClipAdjRel3");
  if(A->dy == NULL) Error(MSG1,"XClipAdjRel3");
  if(A->dz == NULL) Error(MSG1,"XClipAdjRel3");

  j = 0;
  for(i=0; i<B->n; i++){
    if(B->dx[i]>=lower && B->dx[i]<=higher){
      A->dx[j] = B->dx[i];
      A->dy[j] = B->dy[i];
      A->dz[j] = B->dz[i];
      j++;
    }
  }
  DestroyAdjRel3(&B);
}

void     YClipAdjRel3(AdjRel3 *A, int lower, int higher){
  AdjRel3 *B=NULL;
  int i,j,n=0;

  B = CloneAdjRel3(A);
  for(i=0; i<B->n; i++){
    if(B->dy[i]>=lower && B->dy[i]<=higher) n++;
  }
  A->n  = n;
  A->dx = (int *)realloc(A->dx, n*sizeof(int));
  A->dy = (int *)realloc(A->dy, n*sizeof(int));
  A->dz = (int *)realloc(A->dz, n*sizeof(int));
  if(A->dx == NULL) Error(MSG1,"XClipAdjRel3");
  if(A->dy == NULL) Error(MSG1,"XClipAdjRel3");
  if(A->dz == NULL) Error(MSG1,"XClipAdjRel3");

  j = 0;
  for(i=0; i<B->n; i++){
    if(B->dy[i]>=lower && B->dy[i]<=higher){
      A->dx[j] = B->dx[i];
      A->dy[j] = B->dy[i];
      A->dz[j] = B->dz[i];
      j++;
    }
  }
  DestroyAdjRel3(&B);
}

void     ZClipAdjRel3(AdjRel3 *A, int lower, int higher){
  AdjRel3 *B=NULL;
  int i,j,n=0;

  B = CloneAdjRel3(A);
  for(i=0; i<B->n; i++){
    if(B->dz[i]>=lower && B->dz[i]<=higher) n++;
  }
  A->n  = n;
  A->dx = (int *)realloc(A->dx, n*sizeof(int));
  A->dy = (int *)realloc(A->dy, n*sizeof(int));
  A->dz = (int *)realloc(A->dz, n*sizeof(int));
  if(A->dx == NULL) Error(MSG1,"XClipAdjRel3");
  if(A->dy == NULL) Error(MSG1,"XClipAdjRel3");
  if(A->dz == NULL) Error(MSG1,"XClipAdjRel3");

  j = 0;
  for(i=0; i<B->n; i++){
    if(B->dz[i]>=lower && B->dz[i]<=higher){
      A->dx[j] = B->dx[i];
      A->dy[j] = B->dy[i];
      A->dz[j] = B->dz[i];
      j++;
    }
  }
  DestroyAdjRel3(&B);
}

//------------------------------------------

void ClearSceneAdjFrame(Scene *scn, AdjRel3 *A){
  int x,y,z,p;
  int fx,fy,fz;

  FrameSizes3(A, &fx, &fy, &fz);
  
  for(y=0; y<scn->ysize; y++)
    for(z=0; z<scn->zsize; z++)
      for(x=0; x<fx ; x++){
	p = VoxelAddress(scn,x,y,z);
	scn->data[p] = 0;
	p = VoxelAddress(scn,scn->xsize-1-x,y,z);
	scn->data[p] = 0;
      }

  for(x=fx; x<scn->xsize-fx; x++)
    for(z=0; z<scn->zsize; z++)
      for(y=0; y<fy ; y++){
	p = VoxelAddress(scn,x,y,z);
	scn->data[p] = 0;
	p = VoxelAddress(scn,x,scn->ysize-1-y,z);
	scn->data[p] = 0;
      }
  
  for(x=fx; x<scn->xsize-fx; x++)
    for(y=fy; y<scn->ysize-fy; y++)
      for(z=0; z<fz ; z++){
	p = VoxelAddress(scn,x,y,z);
	scn->data[p] = 0;
	p = VoxelAddress(scn,x,y,scn->zsize-1-z);
	scn->data[p] = 0;
      }
}


void FrameSizes3(AdjRel3 *A, 
		 int *sz_x, 
		 int *sz_y, 
		 int *sz_z){
  int i=0;
  *sz_x = INT_MIN;
  *sz_y = INT_MIN;
  *sz_z = INT_MIN;
  for(i=0; i<A->n; i++){
    if(abs(A->dx[i]) > *sz_x) *sz_x = abs(A->dx[i]);
    if(abs(A->dy[i]) > *sz_y) *sz_y = abs(A->dy[i]);
    if(abs(A->dz[i]) > *sz_z) *sz_z = abs(A->dz[i]);
  }
}



float *AdjRel3Distance(AdjRel3 *A){
  float *mg=NULL;
  int i;
  mg = AllocFloatArray(A->n);
  for(i=0; i<A->n; i++){
    mg[i] = sqrt(A->dx[i]*A->dx[i]+
		 A->dy[i]*A->dy[i]+
		 A->dz[i]*A->dz[i]);
  }
  return mg;
}


float *AdjRel3SceneDistance(Scene *scn, AdjRel3 *A){
  float *mg=NULL;
  float dx,dy,dz;
  int i;
  mg = AllocFloatArray(A->n);
  for(i=0; i<A->n; i++){
    dx = A->dx[i]*scn->dx;
    dy = A->dy[i]*scn->dy;
    dz = A->dz[i]*scn->dz;
    mg[i] = sqrt(dx*dx + dy*dy + dz*dz);
  }
  return mg;
}


