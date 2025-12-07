#include "adjacency3.h"

AdjRel3 *CreateAdjRel3(int n)
{
  AdjRel3 *A=NULL;

  A = (AdjRel3 *) calloc(1,sizeof(AdjRel3));
  if (A != NULL){
    A->dx = AllocIntArray(n);
    A->dy = AllocIntArray(n);
    A->dz = AllocIntArray(n);
    A->n  = n;
  } else {
    Error(MSG1,"CreateAdjRel3");
  }

  return(A);
}

void DestroyAdjRel3(AdjRel3 **A)
{
  AdjRel3 *aux;

  aux = *A;
  if (aux != NULL){
    if (aux->dx != NULL) free(aux->dx);
    if (aux->dy != NULL) free(aux->dy);
    if (aux->dz != NULL) free(aux->dz);
    free(aux);
    *A = NULL;
  }   
}


AdjRel3 *CloneAdjRel3(AdjRel3 *A){
  AdjRel3 *C;
  int i;

  C = CreateAdjRel3(A->n);
  for(i=0; i < A->n; i++){
    C->dx[i] = A->dx[i];
    C->dy[i] = A->dy[i];
    C->dz[i] = A->dz[i];
  }

  return C;
}

int ValidAdjacentAddress( Scene *scn, AdjRel3 *A, int p, int i ) {
  Voxel u;
  
  u.x = VoxelX( scn, p ) + A->dx[ i ];
  u.y = VoxelY( scn, p ) + A->dy[ i ];
  u.z = VoxelZ( scn, p ) + A->dz[ i ];
  if( ValidVoxel( scn, u.x, u.y, u.z ) )
    return( VoxelAddress( scn, u.x, u.y, u.z ) );
  return( -1 );
}

int ValidAdjacentAddressMask( Scene *mask, AdjRel3 *A, int p, int i ) {
  Voxel u;
  int q;
  
  u.x = VoxelX( mask, p ) + A->dx[ i ];
  u.y = VoxelY( mask, p ) + A->dy[ i ];
  u.z = VoxelZ( mask, p ) + A->dz[ i ];
  if( ValidVoxel( mask, u.x, u.y, u.z ) ) {
    q = VoxelAddress( mask, u.x, u.y, u.z );
    if( mask->data[ q ] != 0 )
      return( q );
  }
  return( -1 );
}

AdjRel3 *Spheric(float r)
{
  AdjRel3 *A=NULL;
  int i,n,r0,r2,dx,dy,dz,i0=0;

  n=0;
  r0 = (int)r;
  r2  = (int)(r*r + 0.5);
  for(dz=-r0;dz<=r0;dz++)
    for(dy=-r0;dy<=r0;dy++)
      for(dx=-r0;dx<=r0;dx++)
      if(((dx*dx)+(dy*dy)+(dz*dz)) <= r2)
	n++;
	
  A = CreateAdjRel3(n);
  i=0;
  for(dz=-r0;dz<=r0;dz++)
    for(dy=-r0;dy<=r0;dy++)
      for(dx=-r0;dx<=r0;dx++)
	if(((dx*dx)+(dy*dy)+(dz*dz)) <= r2){
	  A->dx[i]=dx;
	  A->dy[i]=dy;
	  A->dz[i]=dz;
	  if ((dx==0)&&(dy==0)&&(dz==0))
	    i0 = i;
	  i++;	  
	}

  /* shift to right and place central voxel at first */
  
  for (i=i0; i > 0; i--) {
    dx = A->dx[i];
    dy = A->dy[i];
    dz = A->dz[i];
    A->dx[i] = A->dx[i-1];
    A->dy[i] = A->dy[i-1];
    A->dz[i] = A->dz[i-1];
    A->dx[i-1] = dx;
    A->dy[i-1] = dy;
    A->dz[i-1] = dz;
  }

  return(A);
}

AdjRel3 *Ellipsoid( float rx, float ry, float rz ) {
  
  AdjRel3 *A=NULL;
  int i,n,dx,dy,dz,i0=0;
  float rx2,ry2,rz2;

  n=0;
  rx2  = rx * rx;
  ry2  = ry * ry;
  rz2  = rz * rz;
  for(dz=-rz;dz<=rz;dz++)
    for(dy=-ry;dy<=ry;dy++)
      for(dx=-rx;dx<=rx;dx++)
	if(((dx*dx)/rx2+(dy*dy)/ry2+(dz*dz)/rz2) <= 1.001)
	n++;
  
  A = CreateAdjRel3(n);

  i=0;
  for(dz=-rz;dz<=rz;dz++)
    for(dy=-ry;dy<=ry;dy++)
      for(dx=-rx;dx<=rx;dx++)
	if(((dx*dx)/rx2+(dy*dy)/ry2+(dz*dz)/rz2) <= 1.001) {
	  A->dx[i]=dx;
	  A->dy[i]=dy;
	  A->dz[i]=dz;
	  if ((dx==0)&&(dy==0)&&(dz==0))
	    i0 = i;
	  i++;	  
	}

  /* shift to right and place central voxel at first */
  
  for (i=i0; i > 0; i--) {
    dx = A->dx[i];
    dy = A->dy[i];
    dz = A->dz[i];
    A->dx[i] = A->dx[i-1];
    A->dy[i] = A->dy[i-1];
    A->dz[i] = A->dz[i-1];
    A->dx[i-1] = dx;
    A->dy[i-1] = dy;
    A->dz[i-1] = dz;
  }

  return(A);
 
}


AdjRel3 *SphericalShell(float inner_radius, 
			float outer_radius){
  AdjRel3 *A=NULL;
  int i,n,dx,dy,dz,r2i,r0o,r2o,d;

  if (outer_radius <= inner_radius){
    Error("outer_radius must be greater than inner_radius","SphericalShell");
    return(NULL);
  }

  n=0;
  r2i  = (int)(inner_radius*inner_radius + 0.5);
  r0o  = (int)outer_radius;
  r2o  = (int)(outer_radius*outer_radius + 0.5);

  for(dz=-r0o;dz<=r0o;dz++){
    for(dy=-r0o;dy<=r0o;dy++){
      for(dx=-r0o;dx<=r0o;dx++){
	d = (dx*dx)+(dy*dy)+(dz*dz);
	if((d <= r2o)&&(d>=r2i))
	  n++;
      }
    }
  }

  A = CreateAdjRel3(n);
  i=0;

  for(dz=-r0o;dz<=r0o;dz++){
    for(dy=-r0o;dy<=r0o;dy++){
      for(dx=-r0o;dx<=r0o;dx++){
	d = (dx*dx)+(dy*dy)+(dz*dz);
	if((d <= r2o)&&(d>=r2i)){
	  A->dx[i]=dx;
	  A->dy[i]=dy;
	  A->dz[i]=dz;
	  i++;
	}
      }
    }
  }
  return(A);
}


int FrameSize3(AdjRel3 *A)
{
  int sz=INT_MIN,i=0;

  for (i=0; i < A->n; i++){
    if (fabs(A->dx[i]) > sz) 
      sz = fabs(A->dx[i]);
    if (fabs(A->dy[i]) > sz) 
      sz = fabs(A->dy[i]);
    if (fabs(A->dz[i]) > sz) 
      sz = fabs(A->dz[i]);
  }
  return(sz);
}

AdjRel3 *Cube(int xsize, int ysize, int zsize)
{
  AdjRel3 *A=NULL;
  int i,dx,dy,dz;

  if (xsize%2 == 0) xsize++;
  if (ysize%2 == 0) ysize++;
  if (zsize%2 == 0) zsize++;
      
  A = CreateAdjRel3(xsize*ysize*zsize);
  i=1;
  for(dz=-zsize/2;dz<=zsize/2;dz++)
    for(dy=-ysize/2;dy<=ysize/2;dy++)
      for(dx=-xsize/2;dx<=xsize/2;dx++)
	if (!((dx == 0)&&(dy == 0)&&(dz == 0))){
	  A->dx[i]=dx;
	  A->dy[i]=dy;
	  A->dz[i]=dz;
	  i++;
	}
  
  
    /* place the central pixel at first */
  A->dx[0] = 0;
  A->dy[0] = 0;
  A->dz[0] = 0;

  return(A);
}


AdjVxl *AdjVoxels(Scene *scn, AdjRel3 *A)
{
  AdjVxl *N;
  int i,xysize;

  xysize = scn->xsize * scn->ysize;

  N = (AdjVxl *) calloc(1,sizeof(AdjVxl));
  if(N != NULL){
    N->dp = AllocIntArray(A->n);
    N->n  = A->n;
    for (i=0; i < N->n; i++)
      N->dp[i] = A->dx[i] + scn->xsize*A->dy[i] + xysize * A->dz[i];
  }else{
    Error(MSG1,"AdjPixels");  
  }

  return(N);
}

void DestroyAdjVxl(AdjVxl **N)
{
  AdjVxl *aux;

  aux = *N;
  if (aux != NULL){
    if (aux->dp != NULL) free(aux->dp);
    free(aux);
    *N = NULL;
  }
}



