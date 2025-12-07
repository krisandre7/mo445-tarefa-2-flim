
#include "scene_addons.h"
#include "arraylist.h"
#include "segmobject.h"
#include <pthread.h>


Scene *GetSceneTransitions3(Scene *scn, AdjRel3 *A){
  Scene *hscn=NULL;
  int p,q,i;
  Voxel u,v;

  hscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (u.z=0; u.z < hscn->zsize; u.z++){
    for (u.y=0; u.y < hscn->ysize; u.y++){
      for (u.x=0; u.x < hscn->xsize; u.x++){
	p = u.x + hscn->tby[u.y] + hscn->tbz[u.z];
	if (scn->data[p] != 0) {
	  for (i=1; i < A->n; i++){
	    v.x = u.x + A->dx[i];
	    v.y = u.y + A->dy[i];
	    v.z = u.z + A->dz[i];
	    if (ValidVoxel(hscn,v.x,v.y,v.z)){
	      q = v.x + hscn->tby[v.y] + hscn->tbz[v.z];
	      if (scn->data[p] != scn->data[q]){
		hscn->data[p] = scn->data[p];
	        break;
	      }
	    }
	  }
	}
      }
    }
  }
  return(hscn);
}


float VoxelEuclideanSquaredDistance(Scene *scn,
				    Voxel u, Voxel v){
  float dx,dy,dz;
  dx = (u.x-v.x)*scn->dx;
  dy = (u.y-v.y)*scn->dy;
  dz = (u.z-v.z)*scn->dz;
  return (dx*dx+dy*dy+dz*dz);
}


Scene *ComputeIntegralScene(Scene *scn){
  Scene *iscn=NULL;
  int p,q,i,j,k;
  int xysize = scn->xsize*scn->ysize;
  bool flag = false;
  long int sum;

  iscn = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  SetVoxelSize(iscn, scn->dx, scn->dy, scn->dz);
  
  for(k=0; k<scn->zsize; k++){
    for(i=0; i<scn->ysize; i++){
      for(j=0; j<scn->xsize; j++){
	p = VoxelAddress(scn,j,i,k);
	sum = scn->data[p];

	if(k-1>=0){
	  q = p-xysize;
	  sum += iscn->data[q];
	}
	if(j-1>=0){
	  q = p-1;
	  sum += iscn->data[q];
	}
	if(j-1>=0 && k-1>=0){
	  q = p-1-xysize;
	  sum -= iscn->data[q];
	}
	if(i-1>=0){
	  q = p-scn->xsize;
	  sum += iscn->data[q];
	}
	if(i-1>=0 && k-1>=0){
	  q = p-scn->xsize-xysize;
	  sum -= iscn->data[q];
	}
	if(j-1>=0 && i-1>=0){
	  q = p-1-scn->xsize;
	  sum -= iscn->data[q];
	}
	if(j-1>=0 && i-1>=0 && k-1>=0){
	  q = p-1-scn->xsize-xysize;
	  sum += iscn->data[q];
	}
	if(sum>INT_MAX) flag = true;
	iscn->data[p] = (int)sum;
      }
    }
  }
  if(flag)
    Warning("Integer overflow","ComputeIntegralScene");

  return iscn;
}


//The dimensions of the window (i.e., xsize,ysize,zsize) 
//should be given in millimeters.
Voxel  FindWindowOfMaximumDensity(Scene *scn, 
				  float xsize, 
				  float ysize, 
				  float zsize){
  Voxel P,u,v;
  Scene *iscn=NULL;
  int dx,dy,dz;
  int q;
  long int sum,max=INT_MIN;

  P.x = P.y = P.z = 0;
  dx = ROUND(xsize/(scn->dx*2.0));
  dy = ROUND(ysize/(scn->dy*2.0));
  dz = ROUND(zsize/(scn->dz*2.0));
  iscn = ComputeIntegralScene(scn);

  for(u.x=0; u.x<scn->xsize; u.x++){
    for(u.y=0; u.y<scn->ysize; u.y++){
      for(u.z=0; u.z<scn->zsize; u.z++){
	v.x = MIN(u.x+dx, scn->xsize-1);
	v.y = MIN(u.y+dy, scn->ysize-1);
	v.z = MIN(u.z+dz, scn->zsize-1);
	q = VoxelAddress(scn,v.x,v.y,v.z);
	sum = iscn->data[q];
	
	v.x = MAX(u.x-dx-1, 0);
	q = VoxelAddress(scn,v.x,v.y,v.z);
	sum -= iscn->data[q];
	
	v.x = MIN(u.x+dx, scn->xsize-1);
	v.y = MAX(u.y-dy-1, 0);
	q = VoxelAddress(scn,v.x,v.y,v.z);
	sum -= iscn->data[q];
	
	v.y = MIN(u.y+dy, scn->ysize-1);
	v.z = MAX(u.z-dz-1, 0);
	q = VoxelAddress(scn,v.x,v.y,v.z);
	sum -= iscn->data[q];
	
	//----------------------------
	v.x = MAX(u.x-dx-1, 0);
	v.y = MIN(u.y+dy, scn->ysize-1);
	v.z = MAX(u.z-dz-1, 0);
	q = VoxelAddress(scn,v.x,v.y,v.z);
	sum += iscn->data[q];
	
	v.x = MAX(u.x-dx-1, 0);
	v.y = MAX(u.y-dy-1, 0);
	v.z = MIN(u.z+dz, scn->zsize-1);
	q = VoxelAddress(scn,v.x,v.y,v.z);
	sum += iscn->data[q];
	
	v.x = MIN(u.x+dx, scn->xsize-1);
	v.y = MAX(u.y-dy-1, 0);
	v.z = MAX(u.z-dz-1, 0);
	q = VoxelAddress(scn,v.x,v.y,v.z);
	sum += iscn->data[q];
	
	v.x = MAX(u.x-dx-1, 0);
	v.y = MAX(u.y-dy-1, 0);
	v.z = MAX(u.z-dz-1, 0);
	q = VoxelAddress(scn,v.x,v.y,v.z);
	sum -= iscn->data[q];
	
	if(sum>max){    
	  max = sum;
	  P = u;
	}
      }
    }
  }
  DestroyScene(&iscn);
  return P;
}


Scene *ChangeOrientationToLPS(Scene *scn, char *ori){
  int Xsrc=0,Ysrc=0,Zsrc=0;
  int xsize=0,ysize=0,zsize=0;
  float dx=1.0,dy=1.0,dz=1.0;
  int p,q,oldx=0,oldy=0,oldz=0;
  Scene *res=NULL;
  Voxel v;
  if      (ori[0]=='L') { Xsrc=+1; xsize=scn->xsize; dx=scn->dx; }
  else if (ori[0]=='R') { Xsrc=-1; xsize=scn->xsize; dx=scn->dx; }
  else if (ori[0]=='P') { Ysrc=+1; ysize=scn->xsize; dy=scn->dx; }
  else if (ori[0]=='A') { Ysrc=-1; ysize=scn->xsize; dy=scn->dx; }
  else if (ori[0]=='S') { Zsrc=+1; zsize=scn->xsize; dz=scn->dx; }
  else if (ori[0]=='I') { Zsrc=-1; zsize=scn->xsize; dz=scn->dx; }
  else{ Error("Invalid orientation","ChangeOrientationToLPS");   }

  if      (ori[1]=='L') { Xsrc=+2; xsize=scn->ysize; dx=scn->dy; }
  else if (ori[1]=='R') { Xsrc=-2; xsize=scn->ysize; dx=scn->dy; }
  else if (ori[1]=='P') { Ysrc=+2; ysize=scn->ysize; dy=scn->dy; }
  else if (ori[1]=='A') { Ysrc=-2; ysize=scn->ysize; dy=scn->dy; }
  else if (ori[1]=='S') { Zsrc=+2; zsize=scn->ysize; dz=scn->dy; }
  else if (ori[1]=='I') { Zsrc=-2; zsize=scn->ysize; dz=scn->dy; }
  else{ Error("Invalid orientation","ChangeOrientationToLPS");   }

  if      (ori[2]=='L') { Xsrc=+3; xsize=scn->zsize; dx=scn->dz; }
  else if (ori[2]=='R') { Xsrc=-3; xsize=scn->zsize; dx=scn->dz; }
  else if (ori[2]=='P') { Ysrc=+3; ysize=scn->zsize; dy=scn->dz; }
  else if (ori[2]=='A') { Ysrc=-3; ysize=scn->zsize; dy=scn->dz; }
  else if (ori[2]=='S') { Zsrc=+3; zsize=scn->zsize; dz=scn->dz; }
  else if (ori[2]=='I') { Zsrc=-3; zsize=scn->zsize; dz=scn->dz; }
  else{ Error("Invalid orientation","ChangeOrientationToLPS");   }

  res= CreateScene(xsize,ysize,zsize);
  SetVoxelSize(res, dx,dy,dz);

  v.z=0; v.y=0;
  for (v.z=0; v.z < zsize; v.z++)
    for (v.y=0; v.y < ysize; v.y++)
      for (v.x=0; v.x < xsize; v.x++) {
	p = v.x + res->tby[v.y] + res->tbz[v.z];
	switch (Xsrc) {
	case +1: oldx=v.x; break;
	case -1: oldx=xsize-v.x-1; break;
	case +2: oldy=v.x; break;
	case -2: oldy=xsize-v.x-1; break;
	case +3: oldz=v.x; break;
	case -3: oldz=xsize-v.x-1; break;
	}
	switch (Ysrc) {
	case +1: oldx=v.y; break;
	case -1: oldx=ysize-v.y-1; break;
	case +2: oldy=v.y; break;
	case -2: oldy=ysize-v.y-1; break;
	case +3: oldz=v.y; break;
	case -3: oldz=ysize-v.y-1; break;
	}
	switch (Zsrc) {
	case +1: oldx=v.z; break;
	case -1: oldx=zsize-v.z-1; break;
	case +2: oldy=v.z; break;
	case -2: oldy=zsize-v.z-1; break;
	case +3: oldz=v.z; break;
	case -3: oldz=zsize-v.z-1; break;
	}
	q = oldx + scn->tby[oldy] + scn->tbz[oldz];
	res->data[p]=scn->data[q];
      }
  
  return res;
}


Scene *LinearInterpCentr3(Scene *scn,float dx,float dy,float dz){
  int value,xsize,ysize,zsize;
  Scene *scene,*tmp;
  Voxel P,Q,R,C; /* previous, current, next voxel and central*/
  float min=FLT_MAX;
  float walked_dist,dist_PQ;

  /* The default voxel sizes of the input scene should be dx=dy=dz=1.0 */

  if ((scn->dx == 0.0) && (scn->dy == 0.0) && (scn->dz == 0.0)) {    
  scn->dx=1.0;
  scn->dy=1.0;
  scn->dz=1.0;
  }

  /* The default voxel sizes of the output scene should be dx=dy=dz=min(dx,dy,dz) */

  if ((dx == 0.0) || (dy == 0.0) || (dz == 0.0)) {
    if (scn->dx < min)
      min = scn->dx;
    if (scn->dy < min)
      min = scn->dy;
    if (scn->dz < min)
      min = scn->dz;
    dx = min; dy = min; dz = min;
    if (min <= 0) {
      fprintf(stderr,"Voxel distance can not be negative.\n");
      exit(-1);
    }
  }

  /* If there is no need for resampling then returns input scene */

  if ((dx == scn->dx) && (dy == scn->dy) && (dz == scn->dz)) {
    scene = CopyScene(scn);
    return (scene);
  } else {
  /* Else the working image is the input image */
    scene = scn;
  }

  /* Resample in x */

  if (dx != scn->dx) {
    xsize = ROUND((float)(scene->xsize)*scene->dx/dx);
    if(xsize%2==0) xsize += 1;
    tmp = CreateScene(xsize, scene->ysize, scene->zsize);
    C.x = (int)xsize/2;
    for(Q.x=0; Q.x < tmp->xsize; Q.x++)
      for(Q.z=0; Q.z < tmp->zsize; Q.z++)
        for(Q.y=0; Q.y < tmp->ysize; Q.y++) {
	  	  
	  walked_dist = (float)(Q.x-C.x) * dx; /* the walked distance so far */
	  
	  /* P is the previous pixel in the original scene */
	  P.x = (int)(walked_dist/scn->dx) + (int)scn->xsize/2; 
	  P.y = Q.y;
	  P.z = Q.z;
	  
	  R.x = P.x + SIGN(walked_dist); /* R is the next pixel in the original
			                    image. Observe that Q is in between P
			                    and R. */
	  R.y = P.y;
	  R.z = P.z;

	  /* the distance between P and Q */
	  dist_PQ =  fabsf(walked_dist - (float)(P.x-scn->xsize/2) * scn->dx);

	  /* interpolation: P --- dPQ --- Q ---- dPR-dPQ ---- R
    
	     I(Q) = (I(P)*(dPR-dPQ) + I(R)*dPQ) / dPR
	     
	  */

	  value = ROUND((( scn->dx - dist_PQ)*(float)VoxelValue(scene,P) + dist_PQ * (float)VoxelValue(scene,R) )/scn->dx);
	  tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
	}
    scene=tmp;
  }

  /* Resample in y */

  if (dy != scn->dy) {
    ysize = ROUND((float)scene->ysize * scn->dy / dy);
    if(ysize%2==0) ysize += 1;
    tmp = CreateScene(scene->xsize, ysize, scene->zsize);
    C.y = (int)ysize/2;
    for(Q.y=0; Q.y < tmp->ysize; Q.y++)
      for(Q.z=0; Q.z < tmp->zsize; Q.z++)
          for(Q.x=0; Q.x < tmp->xsize; Q.x++) {

            walked_dist = (float)(Q.y-C.y) * dy;

            P.x = Q.x;
            P.y = (int)(walked_dist/scn->dy) + (int)scn->ysize/2;
            P.z = Q.z;

            R.x = P.x;
            R.y = P.y + SIGN(walked_dist);
            R.z = P.z;

            dist_PQ =  fabsf(walked_dist - (float)(P.y-scn->ysize/2) * scn->dy);
	    /* comecar a adaptar daqui !! */
            value = ROUND((( (scn->dy - dist_PQ)*(float)VoxelValue(scene,P)) + dist_PQ * (float)VoxelValue(scene,R)) / scn->dy) ;
	    tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
           
          }
    if (scene != scn) {
      DestroyScene(&scene);
    }
    scene=tmp;
  }

  /* Resample in z */

  if (dz != scn->dz) {
    zsize = ROUND((float)scene->zsize * scn->dz / dz);
    if(zsize%2==0) zsize += 1;
    tmp = CreateScene(scene->xsize, scene->ysize, zsize);
    C.z = (int)zsize/2;
    for(Q.z=0; Q.z < tmp->zsize; Q.z++)
        for(Q.y=0; Q.y < tmp->ysize; Q.y++)
          for(Q.x=0; Q.x < tmp->xsize; Q.x++) {

            walked_dist = (float)(Q.z-C.z) * dz;

            P.x = Q.x;
            P.y = Q.y;
            P.z = (int)(walked_dist/scn->dz) + (int)scn->zsize/2;

            R.x = P.x;
            R.y = P.y;
            R.z = P.z + SIGN(walked_dist);

            dist_PQ =  fabsf(walked_dist - (float)(P.z-scn->zsize/2) * scn->dz);

	    value = ROUND((( (scn->dz - dist_PQ)*(float)VoxelValue(scene,P)) + dist_PQ * (float)VoxelValue(scene,R)) / scn->dz) ;
	    tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
	  }
    if (scene != scn) {
      DestroyScene(&scene);
    }
    scene=tmp;
  }
 
  scene->dx=dx;
  scene->dy=dy;
  scene->dz=dz;
  scene->maxval=scn->maxval;
  return(scene);
}


Scene*  LinearRotate3(Scene *scn, 
		      double thx, double thy, double thz, // angles to rotate
		      int cx, int cy, int cz){ // center of the rotation
  int p,Imin = MinimumValue3(scn);
  Voxel u;
  Scene *res;
  RealMatrix *trans1,*rot1,*rot2,*rot3,*trans2,*aux1,*aux2,*inv,*vox;
  trans1 = TranslationMatrix3(-cx,-cy,-cz);
  rot1 = RotationMatrix3(0,thx);
  rot2 = RotationMatrix3(1,thy);
  rot3 = RotationMatrix3(2,thz);
  trans2 = TranslationMatrix3(cx,cy,cz);

  // Compose transform
  aux1 = MultRealMatrix(trans2,rot3);
  aux2 = MultRealMatrix(aux1,rot2);
  DestroyRealMatrix(&aux1);
  aux1 = MultRealMatrix(aux2,rot1);
  DestroyRealMatrix(&aux2);
  aux2 = MultRealMatrix(aux1,trans1);

  inv = InvertRealMatrix(aux2);
  DestroyRealMatrix(&trans1);
  DestroyRealMatrix(&rot1);
  DestroyRealMatrix(&rot2);
  DestroyRealMatrix(&rot3);
  DestroyRealMatrix(&trans2);
  DestroyRealMatrix(&aux2);
  DestroyRealMatrix(&aux1);
  // Applying transform for all voxels
  res = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (u.z=0; u.z < res->zsize; u.z++)
    for (u.y=0; u.y < res->ysize; u.y++)
      for (u.x=0; u.x < res->xsize; u.x++){
	p = u.x + res->tby[u.y] + res->tbz[u.z];
	vox = TransformVoxel(inv,u);
	if ((vox->val[0][0]<=res->xsize-1)&&(vox->val[0][1]<=res->ysize-1)
	    &&(vox->val[0][2]<=res->zsize-1 && vox->val[0][0]>0 && vox->val[0][1]>0 && vox->val[0][2]>0))
	  {
	    res->data[p]=GetVoxelValue_trilinear(scn,vox->val[0][0],vox->val[0][1],vox->val[0][2]);
	  }
	else
	  res->data[p]=Imin;
	DestroyRealMatrix(&vox);
      }
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=scn->dz;
  return res;
}


Scene *ShapeBasedRotate3(Scene *scn,
			 double thx, double thy, double thz,
			 int cx, int cy, int cz){
  Scene *res,*rot,*bin,*dist;
  AdjRel3 *A = Spheric(1.0);
  Curve *c; 
  int k,i,p;

  res = NULL;
  c = SceneLabels(scn);
  for(k=0;k<c->n;k++){
    i = (int)c->X[k];
    /* Avoid background */
    if(i){
      bin = Threshold3(scn,i,i);
      //dist = DistTrans3(bin, A, BOTH, 1);
      dist = Mask2EDT3(bin, A, BOTH, 100, 1);
      rot = LinearRotate3(dist, thx, thy, thz, cx, cy, cz);
      DestroyScene(&bin);
      DestroyScene(&dist);
      bin = Threshold3(rot,0,INT_MAX);
      DestroyScene(&rot);

      if(res==NULL) res = CopyScene(bin);

      for(p=0; p<bin->n; p++)
	if(bin->data[p]>0)
	  res->data[p] = i;
      DestroyScene(&bin);
    }
  }
  res->dx = scn->dx;
  res->dy = scn->dy;
  res->dz = scn->dz;
  DestroyCurve(&c);
  DestroyAdjRel3(&A);

  return res;
}



void    SumMaskCentr3(Scene *acc,
		      Scene *mask){
  int  p,q,n;
  Voxel C,u,v;
  
  ComputeMaskCentroid3(mask, &C);

  n = mask->xsize*mask->ysize*mask->zsize;

  for(p=0; p<n; p++){
    if(mask->data[p]>0){
      u.x = VoxelX(mask, p);
      u.y = VoxelY(mask, p);
      u.z = VoxelZ(mask, p);

      v.x = u.x + (acc->xsize/2 - C.x);
      v.y = u.y + (acc->ysize/2 - C.y);
      v.z = u.z + (acc->zsize/2 - C.z);
      if(ValidVoxel(acc, v.x, v.y, v.z)){
	q = VoxelAddress(acc,v.x,v.y,v.z);
	acc->data[q] += mask->data[p];
      }
      else
	Warning("Accumulator is too small",
		"SumMaskCentr3");
    }
  }
}


void    ComputeMaskCentroid3(Scene *mask, 
			     Voxel *C){
  int x,y,z,p,nelems;

  C->x = C->y = C->z = nelems = 0;
  for(z=0; z<mask->zsize; z++) 
    for(y=0; y<mask->ysize; y++)
      for(x=0; x<mask->xsize; x++){
	p = VoxelAddress(mask, x,y,z);
	if(mask->data[p]>0){
	  C->x += x;
	  C->y += y;
	  C->z += z;
	  nelems++;
	}
      }
  if(nelems==0){
    C->x = mask->xsize/2;
    C->y = mask->ysize/2;
    C->z = mask->zsize/2;
  }
  else{
    C->x /= nelems;
    C->y /= nelems;
    C->z /= nelems;
  }
}



void    ComputeSceneMBB(Scene *scn, 
			Voxel *Vm, 
			Voxel *VM){
  int xsize,ysize,zsize,p,i,j,k,val;
  
  xsize = scn->xsize;
  ysize = scn->ysize;
  zsize = scn->zsize;
  Vm->x = xsize - 1;
  Vm->y = ysize - 1;
  Vm->z = zsize - 1;
  VM->x = 0;
  VM->y = 0;
  VM->z = 0;
  
  for(k=0; k<zsize; k++){
    for(i=0; i<ysize; i++){
      for(j=0; j<xsize; j++){
	p = VoxelAddress(scn,j,i,k);
	val = scn->data[p];

	if(val!=0){
	  if(j < Vm->x) Vm->x = j;
	  if(i < Vm->y) Vm->y = i;
	  if(k < Vm->z) Vm->z = k;
	  if(j > VM->x) VM->x = j;
	  if(i > VM->y) VM->y = i;
	  if(k > VM->z) VM->z = k;
	}
      }
    }
  }
}


Scene  *CopySubScene(Scene *scn,
		     Voxel Vmin,
		     Voxel Vmax){
  return ROI3(scn,
	      Vmin.x, Vmin.y, Vmin.z,
	      Vmax.x, Vmax.y, Vmax.z);
}


void    PasteSubScene(Scene *scn,
		      Scene *sub,
		      Voxel pos){
  int p,q;
  Voxel v;
  for(p=0; p<sub->n; p++){
    v.x = pos.x + VoxelX(sub,p);
    v.y = pos.y + VoxelY(sub,p);
    v.z = pos.z + VoxelZ(sub,p);
    q = VoxelAddress(scn,v.x,v.y,v.z);
    scn->data[q] = sub->data[p];
  }
}


Scene  *AddSceneFrame(Scene *scn, int sx, int sy, int sz, int value){
  Scene *fscn;
  int y, z,*dst,*src,nbytes,offset1, offset2;

  fscn = CreateScene(scn->xsize+(2*sx),scn->ysize+(2*sy), scn->zsize+(2*sz));
  fscn->dx = scn->dx;
  fscn->dy = scn->dy;
  fscn->dz = scn->dz;

  SetScene(fscn,value);
  nbytes = sizeof(int)*scn->xsize;
  
  offset1 = 0;
  offset2 = sx + fscn->tby[sy] + fscn->tbz[sz];
  
  for(z=0; z<scn->zsize; z++){
    src = scn->data+offset1;
    dst = fscn->data+offset2;
    for(y=0; y<scn->ysize; y++){
      memcpy(dst,src,nbytes);
      src += scn->xsize;
      dst += fscn->xsize;
    }
    offset1 += scn->xsize*scn->ysize;
    offset2 += fscn->xsize*fscn->ysize;
  }
  return(fscn);
}


Scene  *RemSceneFrame(Scene *fscn, int sx, int sy, int sz){
  Scene *scn;
  int y,z,*dst,*src,nbytes,offset;

  scn = CreateScene(fscn->xsize-(2*sx),fscn->ysize-(2*sy),fscn->zsize-(2*sz));
  scn->dx = fscn->dx;
  scn->dy = fscn->dy;
  scn->dz = fscn->dz;

  nbytes = sizeof(int)*scn->xsize;  
  offset = sx + fscn->tby[sy] + fscn->tbz[sz];
  src = fscn->data+offset;
  dst = scn->data;
  for(z=0; z<scn->zsize; z++,src+=2*sy*fscn->xsize){
    for(y=0; y<scn->ysize; y++,src+=fscn->xsize,dst+=scn->xsize){
      memcpy(dst,src,nbytes);
    }
  }
  return(scn);
}


Scene *WeightedMean3(Scene *scn1, Scene *scn2, float w,
		     int min1, int max1, 
		     int min2, int max2,
		     int nmin, int nmax){
  Scene *scn=NULL;
  float v1,v2;
  int p;
  
  if(scn1->n!=scn2->n) return NULL;
  scn = CreateScene(scn1->xsize, scn1->ysize, scn1->zsize);
  for(p=0; p<scn1->n; p++){
    v1 = (float)IntegerNormalize(scn1->data[p],
				 min1, max1,
				 nmin, nmax);
    v2 = (float)IntegerNormalize(scn2->data[p],
				 min2, max2,
				 nmin, nmax);
    scn->data[p] = ROUND(w*v1+(1.0-w)*v2);
  }
  return scn;
}


void   WeightedMean3inplace(Scene *scn1, Scene *scn2, 
			    float w,
			    int min1, int max1, 
			    int min2, int max2,
			    int nmin, int nmax){
  float v1,v2;
  int p;
  
  if(scn1->n!=scn2->n) return;
  for(p=0; p<scn1->n; p++){
    v1 = (float)IntegerNormalize(scn1->data[p],
				 min1, max1,
				 nmin, nmax);
    v2 = (float)IntegerNormalize(scn2->data[p],
				 min2, max2,
				 nmin, nmax);
    scn1->data[p] = ROUND(w*v1+(1.0-w)*v2);
  }
}



void  SceneNormalize(Scene *scn,
		     int omin,int omax,
		     int nmin,int nmax){
  int p;
  for(p=0; p<scn->n; p++)
    scn->data[p] = IntegerNormalize(scn->data[p],
				    omin, omax,
				    nmin, nmax);
}




void SetSceneFrame(Scene *scn, int sz, int value){
  int x,y,z,p;

  for(y=0; y<scn->ysize; y++)
    for(z=0; z<scn->zsize; z++)
      for(x=0; x<sz ; x++){
	p = VoxelAddress(scn,x,y,z);
	scn->data[p] = value;
	p = VoxelAddress(scn,scn->xsize-1-x,y,z);
	scn->data[p] = value;
      }

  for(x=sz; x<scn->xsize-sz; x++)
    for(z=0; z<scn->zsize; z++)
      for(y=0; y<sz ; y++){
	p = VoxelAddress(scn,x,y,z);
	scn->data[p] = value;
	p = VoxelAddress(scn,x,scn->ysize-1-y,z);
	scn->data[p] = value;
      }

  for(x=sz; x<scn->xsize-sz; x++)
    for(y=sz; y<scn->ysize-sz; y++)
      for(z=0; z<sz ; z++){
	p = VoxelAddress(scn,x,y,z);
	scn->data[p] = value;
	p = VoxelAddress(scn,x,y,scn->zsize-1-z);
	scn->data[p] = value;
      }
}


void   CopySceneFrame(Scene *dest, Scene *src, int sz){
  int x,y,z,p;

  if(dest->xsize!=src->xsize ||
     dest->ysize!=src->ysize ||
     dest->zsize!=src->zsize){
    Warning("Incompatible sizes","CopySceneFrame");
    return;
  }

  for(y=0; y<src->ysize; y++)
    for(z=0; z<src->zsize; z++)
      for(x=0; x<sz ; x++){
	p = VoxelAddress(src,x,y,z);
	dest->data[p] = src->data[p];
	p = VoxelAddress(src,src->xsize-1-x,y,z);
	dest->data[p] = src->data[p];
      }

  for(x=sz; x<src->xsize-sz; x++)
    for(z=0; z<src->zsize; z++)
      for(y=0; y<sz ; y++){
	p = VoxelAddress(src,x,y,z);
	dest->data[p] = src->data[p];
	p = VoxelAddress(src,x,src->ysize-1-y,z);
	dest->data[p] = src->data[p];
      }

  for(x=sz; x<src->xsize-sz; x++)
    for(y=sz; y<src->ysize-sz; y++)
      for(z=0; z<sz ; z++){
	p = VoxelAddress(src,x,y,z);
	dest->data[p] = src->data[p];
	p = VoxelAddress(src,x,y,src->zsize-1-z);
	dest->data[p] = src->data[p];
      }
}



Scene *InterpScene2Isotropic(Scene *scn){
  float d;
  d = MIN(scn->dx, MIN(scn->dy, scn->dz));
  if(scn->dx!=scn->dy || scn->dx!=scn->dz || scn->dy!=scn->dz)
    return LinearInterp(scn,d,d,d);
  else
    return CopyScene(scn);
}



Scene *BIA_InterpScene2Isotropic(Scene *scn, ArrayList *segmobjs){
  float d;
  d = MIN(scn->dx, MIN(scn->dy, scn->dz));
  if(scn->dx!=scn->dy || scn->dx!=scn->dz || scn->dy!=scn->dz)
    return BIA_LinearInterp(scn,d,d,d,segmobjs);
  else
    return CopyScene(scn);
}


Scene *BIA_LinearInterp(Scene *scn,float dx,float dy,float dz, ArrayList *segmobjs){
  Scene *res = LinearInterp(scn,dx,dy,dz);
  if (res!=NULL) {
    SegmObject *obj=NULL;
    int i;
    int n = segmobjs->n;
    for (i=0; i < n; i++) {
      obj = (SegmObject *)GetArrayListElement(segmobjs, i);
      Scene *mask1 = CreateScene(scn->xsize,scn->ysize,scn->zsize);
      mask1->dx=scn->dx;
      mask1->dy=scn->dy;
      mask1->dz=scn->dz;
      CopyBMap2SceneMask(mask1,obj->mask);
      Scene *mask2 = LinearInterp(mask1,dx,dy,dz);
      DestroyScene(&mask1);
      BMap *newbmap;
      newbmap = SceneMask2BMap(mask2);
      DestroyScene(&mask2);
      BMapDestroy(obj->mask);
      obj->mask = newbmap;
    }    
  }
  return res;
}







int    GetRadiometricRes3(Scene *scn){
  int bpp, maxval;

  maxval = MaximumValue3(scn);
  bpp = 0;
  while(maxval > 0){
    maxval >>= 1;
    bpp++;
  }
  
  return bpp;
}


/* Compute mean and stdev of the input scene */
void ComputeDescriptiveStatistics(Scene *scn, float *mean, float *stdev){
  int n, p;

  n = scn->xsize*scn->ysize*scn->zsize;
  *mean  = 0.0;
  for(p=0; p<n; p++)
    *mean += scn->data[p];
  *mean /= n;

  *stdev = 0.0;
  for(p=0; p<n; p++)
    *stdev += (scn->data[p] - *mean)*(scn->data[p] - *mean);
  *stdev = sqrt(*stdev/n);
}



float *SceneIntensity2Gaussian(Scene *scn, int mean,
			       float stdev_lower, 
			       float stdev_higher){
  int i,nbins = MaximumValue3(scn)+1;
  float *weight=NULL,d;

  weight = AllocFloatArray(nbins);
  for(i=0; i<nbins; i++){
    d = (float)abs(mean-i);
    if(i<=mean)
      weight[i] = (float)exp(-d*d/(2.0*stdev_lower*stdev_lower));
    else
      weight[i] = (float)exp(-d*d/(2.0*stdev_higher*stdev_higher));
  }
  return weight;
}


//--------------------------------------------------

Scene *FastLinearInterpCentr3(Scene *scn,float dx,float dy,float dz){
  ArgLinearInterpCentr3 arg;
  ArgLinearInterpCentr3 args[8];
  int i,nprocs;
  int first,last,nelems,de;
  pthread_t thread_id[8];
  int iret[8];
  float min=FLT_MAX;
  int xsize,ysize,zsize;
  Scene *scene;

  nprocs = GetNumberOfProcessors();
  //printf("nprocs: %d\n",nprocs);

  if(nprocs<=1) return LinearInterpCentr3(scn,dx,dy,dz);
  if(nprocs>=8) nprocs = 8;

  //---------------------------------

  //The default voxel sizes of the input scene should be dx=dy=dz=1.0
  if((scn->dx == 0.0) && (scn->dy == 0.0) && (scn->dz == 0.0)) {    
    scn->dx=1.0;
    scn->dy=1.0;
    scn->dz=1.0;
  }

  //The default voxel sizes of the output scene should be dx=dy=dz=min(dx,dy,dz)
  if((dx == 0.0) || (dy == 0.0) || (dz == 0.0)) {
    if(scn->dx < min) min = scn->dx;
    if(scn->dy < min) min = scn->dy;
    if(scn->dz < min) min = scn->dz;
    dx = min; dy = min; dz = min;
    if(min <= 0) {
      fprintf(stderr,"Voxel distance can not be negative.\n");
      exit(-1);
    }
  }

  // If there is no need for resampling then returns input scene
  if((dx == scn->dx) && (dy == scn->dy) && (dz == scn->dz)) {
    return (CopyScene(scn));
  } 

  scene = scn;
  // Resample in x
  if(dx != scn->dx){
    xsize = ROUND((float)(scn->xsize)*scn->dx/dx);
    if(xsize%2==0) xsize += 1;
    arg.scn  = scn;
    arg.iscn = CreateScene(xsize, scn->ysize, scn->zsize);
    (arg.iscn)->dx=dx;
    
    first  = 0;
    last   = (arg.iscn)->xsize-1;
    nelems = last-first+1; 
    de     = nelems/nprocs;
    arg.i  = NIL;
    arg.j  = first-1;
    for(i=0; i<nprocs; i++){
      args[i] = arg;
      
      args[i].i = arg.j+1;
      if(i<nprocs-1) args[i].j = args[i].i+(de-1);
      else           args[i].j = last;

      //Create independent threads each of which will execute function
      iret[i] = pthread_create(&thread_id[i], NULL, 
			       ThreadLinearInterpXCentr3,
			       (void*)&args[i]);
      arg = args[i];
    }
    
    //Wait till threads are complete before main continues.
    for(i=0; i<nprocs; i++){
      pthread_join(thread_id[i], NULL);
    }
    scene = arg.iscn;
  }

  //---------------------------------

  // Resample in y
  if(dy != scn->dy) {
    ysize = ROUND((float)scene->ysize * scn->dy / dy);
    if(ysize%2==0) ysize += 1;
    arg.scn  = scene;
    arg.iscn = CreateScene(scene->xsize, ysize, scene->zsize);
    (arg.iscn)->dy=dy;

    first  = 0;
    last   = (arg.iscn)->ysize-1;
    nelems = last-first+1; 
    de     = nelems/nprocs;
    arg.i  = NIL;
    arg.j  = first-1;
    for(i=0; i<nprocs; i++){
      args[i] = arg;
      
      args[i].i = arg.j+1;
      if(i<nprocs-1) args[i].j = args[i].i+(de-1);
      else           args[i].j = last;

      //Create independent threads each of which will execute function
      iret[i] = pthread_create(&thread_id[i], NULL, 
			       ThreadLinearInterpYCentr3,
			       (void*)&args[i]);
      arg = args[i];
    }
    
    //Wait till threads are complete before main continues.
    for(i=0; i<nprocs; i++){
      pthread_join(thread_id[i], NULL);
    }
    if(scene != scn) DestroyScene(&scene);
    scene = arg.iscn;
  }

  //---------------------------------

  // Resample in z
  if(dz != scn->dz) {
    zsize = ROUND((float)scene->zsize * scn->dz / dz);
    if(zsize%2==0) zsize += 1;
    arg.scn  = scene;
    arg.iscn = CreateScene(scene->xsize, scene->ysize, zsize);
    (arg.iscn)->dz=dz;

    first  = 0;
    last   = (arg.iscn)->zsize-1;
    nelems = last-first+1; 
    de     = nelems/nprocs;
    arg.i  = NIL;
    arg.j  = first-1;
    for(i=0; i<nprocs; i++){
      args[i] = arg;
      
      args[i].i = arg.j+1;
      if(i<nprocs-1) args[i].j = args[i].i+(de-1);
      else           args[i].j = last;

      //Create independent threads each of which will execute function
      iret[i] = pthread_create(&thread_id[i], NULL,
			       ThreadLinearInterpZCentr3,
			       (void*)&args[i]);
      arg = args[i];
    }
    
    //Wait till threads are complete before main continues.
    for(i=0; i<nprocs; i++){
      pthread_join(thread_id[i], NULL);
    }
    if(scene != scn) DestroyScene(&scene);
    scene = arg.iscn;
  }

  scene->dx=dx;
  scene->dy=dy;
  scene->dz=dz;
  scene->maxval=scn->maxval;
  return(scene);
}


void  *ThreadLinearInterpXCentr3(void *arg){
  ArgLinearInterpCentr3 *p_arg;
  int value;
  Scene *scn,*tmp;
  Voxel P,Q,R,C; //previous, current, next voxel and central
  float walked_dist,dist_PQ,dx;

  // Resample in x
  p_arg = (ArgLinearInterpCentr3 *)arg;
  scn = p_arg->scn;
  tmp = p_arg->iscn;
  dx  = tmp->dx;
  C.x = (int)tmp->xsize/2;
  for(Q.x=p_arg->i; Q.x<=p_arg->j; Q.x++){
    //the walked distance so far
    walked_dist = (float)(Q.x-C.x) * dx; 
    // P is the previous pixel in the original scene
    P.x = (int)(walked_dist/scn->dx) + (int)scn->xsize/2; 
    // R is the next pixel in the original
    //image. Observe that Q is in between P
    //and R.
    R.x = P.x + SIGN(walked_dist); 
    // the distance between P and Q
    dist_PQ =  fabsf(walked_dist - (float)(P.x-scn->xsize/2) * scn->dx);

    for(Q.z=0; Q.z < tmp->zsize; Q.z++){
      for(Q.y=0; Q.y < tmp->ysize; Q.y++) {
	P.y = Q.y;
	P.z = Q.z;
	R.y = P.y;
	R.z = P.z;

	// interpolation: P --- dPQ --- Q ---- dPR-dPQ ---- R
	//   I(Q) = (I(P)*(dPR-dPQ) + I(R)*dPQ) / dPR

	value = ROUND(((scn->dx - dist_PQ)*(float)VoxelValue(scn,P) + 
		       dist_PQ * (float)VoxelValue(scn,R) )/scn->dx);
	tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
      }
    }
  }
  return NULL;
}


void  *ThreadLinearInterpYCentr3(void *arg){
  ArgLinearInterpCentr3 *p_arg;
  int value;
  Scene *scn,*tmp;
  Voxel P,Q,R,C; //previous, current, next voxel and central
  float walked_dist,dist_PQ,dy;

  // Resample in y
  p_arg = (ArgLinearInterpCentr3 *)arg;
  scn = p_arg->scn;
  tmp = p_arg->iscn;
  dy  = tmp->dy;
  C.y = (int)tmp->ysize/2;
  for(Q.y=p_arg->i; Q.y<=p_arg->j; Q.y++){
    walked_dist = (float)(Q.y-C.y) * dy;
    P.y = (int)(walked_dist/scn->dy) + (int)scn->ysize/2;
    R.y = P.y + SIGN(walked_dist);
    dist_PQ = fabsf(walked_dist - (float)(P.y-scn->ysize/2) * scn->dy);
    for(Q.z=0; Q.z < tmp->zsize; Q.z++){
      for(Q.x=0; Q.x < tmp->xsize; Q.x++) {
	P.x = Q.x;
	P.z = Q.z;
	R.x = P.x;
	R.z = P.z;

	value = ROUND((( (scn->dy - dist_PQ)*(float)VoxelValue(scn,P)) + 
		       dist_PQ * (float)VoxelValue(scn,R)) / scn->dy);
	tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
      }
    }
  }
  return NULL;
}


void  *ThreadLinearInterpZCentr3(void *arg){
  ArgLinearInterpCentr3 *p_arg;
  int value;
  Scene *scn,*tmp;
  Voxel P,Q,R,C; //previous, current, next voxel and central
  float walked_dist,dist_PQ,dz;

  // Resample in z
  p_arg = (ArgLinearInterpCentr3 *)arg;
  scn = p_arg->scn;
  tmp = p_arg->iscn;
  dz  = tmp->dz;
  C.z = (int)tmp->zsize/2;

  for(Q.z=p_arg->i; Q.z<=p_arg->j; Q.z++){
    walked_dist = (float)(Q.z-C.z) * dz;
    P.z = (int)(walked_dist/scn->dz) + (int)scn->zsize/2;
    R.z = P.z + SIGN(walked_dist);
    dist_PQ =  fabsf(walked_dist - (float)(P.z-scn->zsize/2) * scn->dz);

    for(Q.y=0; Q.y < tmp->ysize; Q.y++){
      for(Q.x=0; Q.x < tmp->xsize; Q.x++) {
	P.x = Q.x;
	P.y = Q.y;
	R.x = P.x;
	R.y = P.y;
	
	value = ROUND((( (scn->dz - dist_PQ)*(float)VoxelValue(scn,P)) + 
		       dist_PQ * (float)VoxelValue(scn,R)) / scn->dz) ;
	tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
      }
    }
  }
  return NULL;
}




