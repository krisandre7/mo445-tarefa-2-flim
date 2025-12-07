
#include "cloud3.h"


Cloud3 *CreateCloud3(int n){
  Cloud3 *C=NULL;

  C = (Cloud3 *) calloc(1,sizeof(Cloud3));
  if(C != NULL){
    if(n>0){
      C->dx = AllocIntArray(n);
      C->dy = AllocIntArray(n);
      C->dz = AllocIntArray(n);
      C->n  = n;
    }
    else{
      C->dx = NULL; 
      C->dy = NULL; 
      C->dz = NULL;
      C->n  = 0;
    }
    C->dp = NULL;
    C->xsize  = -1;
    C->xysize = -1;
    C->max[0] = INT_MIN;
    C->max[1] = INT_MIN;
    C->max[2] = INT_MIN;
    C->min[0] = INT_MAX;
    C->min[1] = INT_MAX;
    C->min[2] = INT_MAX;
  }
  else
    Error(MSG1,"CreateCloud3");

  return(C);
}


void DestroyCloud3(Cloud3 **cloud){
  Cloud3 *aux;

  aux = *cloud;
  if(aux != NULL){
    if(aux->dx != NULL) free(aux->dx);
    if(aux->dy != NULL) free(aux->dy);
    if(aux->dz != NULL) free(aux->dz);
    if(aux->dp != NULL) free(aux->dp);
    free(aux);
    *cloud = NULL;
  }   
}


Cloud3 *CloneCloud3(Cloud3 *cloud){
  Cloud3 *C;
  int i;

  C = CreateCloud3(cloud->n);
  for(i=0; i < cloud->n; i++){
    C->dx[i] = cloud->dx[i];
    C->dy[i] = cloud->dy[i];
    C->dz[i] = cloud->dz[i];
  }

  return C;
}


Cloud3 *MergeCloud3(Cloud3 *c1, Cloud3 *c2){
  int dx_max,dy_max,dz_max;
  int dx_min,dy_min,dz_min;
  Scene *mask=NULL;
  Cloud3 *c3=NULL;
  Voxel v;

  RefreshCloud3Limits(c1);
  RefreshCloud3Limits(c2);
  dx_max = MAX( c1->max[0], c2->max[0] );
  dy_max = MAX( c1->max[1], c2->max[1] );
  dz_max = MAX( c1->max[2], c2->max[2] );
  dx_min = MIN( c1->min[0], c2->min[0] );
  dy_min = MIN( c1->min[1], c2->min[1] );
  dz_min = MIN( c1->min[2], c2->min[2] );

  mask = CreateScene(abs(dx_min)+dx_max+1,
		     abs(dy_min)+dy_max+1,
		     abs(dz_min)+dz_max+1);
  v.x = abs(dx_min);
  v.y = abs(dy_min);
  v.z = abs(dz_min);
  DrawCloud32Scene(c1, mask, v, 1);
  DrawCloud32Scene(c2, mask, v, 1);
  c3 = Mask2Cloud3(mask, v);
  DestroyScene(&mask);

  return c3;
}


Cloud3 *AdjRel2Cloud3(AdjRel3 *A){
  Cloud3 *C;
  int i;

  C = CreateCloud3(A->n);
  for(i=0; i < A->n; i++){
    C->dx[i] = A->dx[i];
    C->dy[i] = A->dy[i];
    C->dz[i] = A->dz[i];
  }

  return C;
}


Cloud3 *Mask2Cloud3(Scene *mask,
		    Voxel Ref){
  Cloud3 *C;
  Voxel v;
  int i,p,n,nelems=0;

  n = mask->xsize*mask->ysize*mask->zsize;
  for(p=0; p<n; p++)
    if(mask->data[p]>0)
      nelems++;

  C = CreateCloud3(nelems);

  i = 0;
  for(p=0; p<n; p++){
    if(mask->data[p]>0){
      v.x = VoxelX(mask, p);
      v.y = VoxelY(mask, p);
      v.z = VoxelZ(mask, p);
      C->dx[i] = v.x - Ref.x;
      C->dy[i] = v.y - Ref.y;
      C->dz[i] = v.z - Ref.z;
      i++;
    }
  }

  return C;
}


Scene *Cloud32Mask(Cloud3 *cloud){
  int i,dx,dy,dz,dxmax,dymax,dzmax;
  Scene *mask;
  Voxel v;

  dxmax = dymax = dzmax = 0;
  for(i=0; i<cloud->n; i++){
    dx = abs(cloud->dx[i]);
    dy = abs(cloud->dy[i]);
    dz = abs(cloud->dz[i]);
    if(dx>dxmax) dxmax = dx;
    if(dy>dymax) dymax = dy;
    if(dz>dzmax) dzmax = dz;
  }

  mask = CreateScene(dxmax*2+1, 
		     dymax*2+1,
		     dzmax*2+1);
  v.x = dxmax;
  v.y = dymax;
  v.z = dzmax;
  DrawCloud32Scene(cloud, mask, v, 1);

  return mask;
}


void   DrawCloud32Scene(Cloud3 *cloud,
			Scene *scn,
			Voxel u,
			int val){
  Voxel v;
  int i,p;

  for(i=0; i<cloud->n; i++){
    v.x = u.x + cloud->dx[i];
    v.y = u.y + cloud->dy[i];
    v.z = u.z + cloud->dz[i];

    if(ValidVoxel(scn,v.x,v.y,v.z)){
      p = VoxelAddress(scn,v.x,v.y,v.z);
      scn->data[p] = val;
    }
  }
}


void   DrawOptCloud32Scene(Cloud3 *cloud,
			   Scene *scn,
			   int p, int val){
  int i,q;

  OptimizeCloud3(cloud, scn);
  for(i=0; i<cloud->n; i++){
    q = p + cloud->dp[i];
    scn->data[q] = val;
  }
}


void    OptimizeCloud3(Cloud3 *cloud,
		       Scene *scn){
  Cloud3 *C = cloud;
  int xsize,xysize,i;

  xsize  = scn->xsize;
  xysize = scn->xsize * scn->ysize;

  if(C->xsize==xsize && C->xysize==xysize && C->dp!=NULL)
    return;
  
  if(C->dp!=NULL) free(C->dp);
  C->dp = AllocIntArray(C->n);
  
  for(i=0; i<C->n; i++)
    C->dp[i] = C->dx[i] + xsize*C->dy[i] + xysize*C->dz[i];

  C->xsize  = xsize;
  C->xysize = xysize;
}


void    RefreshCloud3Limits(Cloud3 *cloud){
  int dx_min, dy_min, dz_min;
  int dx_max, dy_max, dz_max;
  int i,dx,dy,dz;

  dx_min = dy_min = dz_min = INT_MAX;
  dx_max = dy_max = dz_max = INT_MIN;
  for(i=0; i<cloud->n; i++){
    dx = cloud->dx[i];
    dy = cloud->dy[i];
    dz = cloud->dz[i];
    if(dx > dx_max) dx_max = dx;
    if(dy > dy_max) dy_max = dy;
    if(dz > dz_max) dz_max = dz;
    if(dx < dx_min) dx_min = dx;
    if(dy < dy_min) dy_min = dy;
    if(dz < dz_min) dz_min = dz;
  }
  cloud->max[0] = dx_max;
  cloud->max[1] = dy_max;
  cloud->max[2] = dz_max;
  cloud->min[0] = dx_min;
  cloud->min[1] = dy_min;
  cloud->min[2] = dz_min;
}


void    GetCloud3Limits(Cloud3 *cloud,
			int *dx_min, int *dy_min, int *dz_min,
			int *dx_max, int *dy_max, int *dz_max){
  if(cloud->max[0]<cloud->min[0])
    RefreshCloud3Limits(cloud);
  
  *dx_max = cloud->max[0];
  *dy_max = cloud->max[1];
  *dz_max = cloud->max[2];
  *dx_min = cloud->min[0];
  *dy_min = cloud->min[1];
  *dz_min = cloud->min[2];
}


int     Cloud3FitInside(Cloud3 *cloud,
			Voxel vx,
			Scene *scn,
			int sz){
  int max[3],min[3];
  GetCloud3Limits(cloud,
		  &min[0], &min[1], &min[2],
		  &max[0], &max[1], &max[2]);

  if(vx.x < sz-min[0] ||
     vx.y < sz-min[1] ||
     vx.z < sz-min[2] ||
     vx.x+max[0]>=scn->xsize-sz ||
     vx.y+max[1]>=scn->ysize-sz ||
     vx.z+max[2]>=scn->zsize-sz)
    return 0;
  else
    return 1;
}



float   MeanInsideCloud3(Cloud3 *cloud,
			 Voxel vx,
			 Scene *scn){
  Voxel v;
  float sum=0.0;
  int i,p;

  for(i=0; i<cloud->n; i++){
    v.x = vx.x + cloud->dx[i];
    v.y = vx.y + cloud->dy[i];
    v.z = vx.z + cloud->dz[i];

    if(ValidVoxel(scn,v.x,v.y,v.z)){
      p = VoxelAddress(scn,v.x,v.y,v.z);
      sum += scn->data[p];
    }
  }
  return (sum/cloud->n);
}



float   SumInsideCloud3(Cloud3 *cloud,
			Voxel vx,
			Scene *scn){
  Voxel v;
  float sum=0.0;
  int i,p;

  for(i=0; i<cloud->n; i++){
    v.x = vx.x + cloud->dx[i];
    v.y = vx.y + cloud->dy[i];
    v.z = vx.z + cloud->dz[i];

    if(ValidVoxel(scn,v.x,v.y,v.z)){
      p = VoxelAddress(scn,v.x,v.y,v.z);
      sum += scn->data[p];
    }
  }
  return sum;
}



float   SumInsideCloudMask3(Cloud3 *cloud,
			    Voxel vx,
			    Scene *scn,
			    Scene *mask){
  Voxel v;
  float sum=0.0;
  int i,p;

  for(i=0; i<cloud->n; i++){
    v.x = vx.x + cloud->dx[i];
    v.y = vx.y + cloud->dy[i];
    v.z = vx.z + cloud->dz[i];

    if(ValidVoxel(scn,v.x,v.y,v.z)){
      p = VoxelAddress(scn,v.x,v.y,v.z);
      if(mask->data[p]>0)
	sum += scn->data[p];
    }
  }
  return sum;
}





