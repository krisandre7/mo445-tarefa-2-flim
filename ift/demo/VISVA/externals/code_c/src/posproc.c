

#include "posproc.h"


void ClearBinBelowThreshold(Scene *scn, 
			    Scene *bin,
			    int T){
  int p;
  for(p=0; p<scn->n; p++)
    if(scn->data[p]<T)
      bin->data[p] = 0;

  SelectLargestComp(bin);
}


void ClearPeripheralBinBelowThreshold(Scene *scn, 
				      Scene *bin,
				      float r, int T){
  AdjRel3 *A6 = Spheric(1.0);
  AdjRel3 *A = Spheric(r);
  Scene *border=NULL,*choles=NULL;
  Voxel u,v,COG;
  int p,q,i;
  float r2 = 40.0*40.0; //35.0*35.0;
  FIFOQ *Q;

  ComputeMaskCentroid3(bin, &COG);
  Q = FIFOQNew(scn->n);
  choles = CloseHoles3(bin);
  border = GetBorder3(choles, A6);
  DestroyScene(&choles);

  for(u.z=0; u.z<scn->zsize; u.z++){
    for(u.y=0; u.y<scn->ysize; u.y++){
      for(u.x=0; u.x<scn->xsize; u.x++){
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	if(border->data[p]==0) continue;

	for(i=0; i<A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if(!ValidVoxel(scn,v.x,v.y,v.z)) continue;
	  if(VoxelEuclideanSquaredDistance(scn, v,COG)<=r2)
	    continue;
	  q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	  if(bin->data[q]>0 && scn->data[q]<T){
	    bin->data[q] = 0;
	    FIFOQPush(Q, q);
	  }
	}
      }
    }
  }

  while(!FIFOQEmpty(Q)){
    p = FIFOQPop(Q);
    u.x = VoxelX(scn, p);
    u.y = VoxelY(scn, p);
    u.z = VoxelZ(scn, p);
    for(i=0; i<A6->n; i++){
      v.x = u.x + A6->dx[i];
      v.y = u.y + A6->dy[i];
      v.z = u.z + A6->dz[i];
      if(!ValidVoxel(scn,v.x,v.y,v.z)) continue;
      if(VoxelEuclideanSquaredDistance(scn, v,COG)<=r2)
	continue;
      q = v.x + scn->tby[v.y] + scn->tbz[v.z];
      if(bin->data[q]>0 && scn->data[q]<T){
	bin->data[q] = 0;
	FIFOQPush(Q, q);
      }
    }
  }

  SelectLargestComp(bin);
  
  DestroyScene(&border);
  DestroyAdjRel3(&A);
  DestroyAdjRel3(&A6);
  FIFOQDestroy(Q);
}



Scene *SmoothObjInterfaces3(Scene *label, float r){
  AdjRel3 *A = Spheric(r);
  AdjVxl  *N;
  int fx,fy,fz,dp;
  int p,q,i,Lmax,l,lmax;
  int *frequency;
  Scene *flabel=NULL;

  flabel = CreateScene(label->xsize, 
		       label->ysize, 
		       label->zsize);
  SetVoxelSize(flabel, label->dx, label->dy, label->dz);

  Lmax = MaximumValue3(label);
  frequency = (int *)calloc(Lmax+1,sizeof(int));

  N = AdjVoxels(label, A);
  FrameSizes3(A, &fx, &fy, &fz);
  dp  = fx*1;
  dp += fy*label->xsize;
  dp += fz*label->xsize*label->ysize;

  for(p=dp; p<label->n-dp; p++){
    if(label->data[p]==0) continue;

    memset(frequency, 0, (Lmax+1)*sizeof(int));

    for(i=0; i<N->n; i++){
      q = p + N->dp[i];
      frequency[label->data[q]]++;
    }
    lmax = label->data[p];
    for(l=1; l<=Lmax; l++){
      if(frequency[l]>frequency[lmax])
	lmax = l;
    }
    flabel->data[p] = lmax;
  }
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&N);
  free(frequency);

  return flabel;
}



void ModeFilterLabel3(Scene *label, float r){
  AdjRel3 *A = Spheric(r);
  AdjVxl  *N;
  int fx,fy,fz,dp;
  int p,q,i,Lmax,l,lmax;
  int *frequency;
  Scene *sub=NULL,*mode=NULL,*tmp=NULL;
  BoxSelection3 box;

  box = BoxSelectionByMBB3(label);
  sub = CopySubSceneInBoxSelection3(label, box);
  FrameSizes3(A, &fx, &fy, &fz);
  tmp = AddSceneFrame(sub, fx, fy, fz, 0);
  DestroyScene(&sub);
  sub = tmp;
  mode = CopyScene(sub);

  Lmax = MaximumValue3(sub);
  frequency = (int *)calloc(Lmax+1,sizeof(int));

  N = AdjVoxels(sub, A);
  dp  = fx*1;
  dp += fy*sub->xsize;
  dp += fz*sub->xsize*sub->ysize;

  for(p=dp; p<sub->n-dp; p++){

    memset(frequency, 0, (Lmax+1)*sizeof(int));

    for(i=0; i<N->n; i++){
      q = p + N->dp[i];
      frequency[sub->data[q]]++;
    }
    lmax = sub->data[p];
    for(l=0; l<=Lmax; l++){
      if(frequency[l]>frequency[lmax])
	lmax = l;
    }
    mode->data[p] = lmax;
  }
  DestroyScene(&sub);

  sub = RemSceneFrame(mode, fx, fy, fz);
  PasteSubScene(label, sub, box.v1);

  DestroyScene(&sub);
  DestroyScene(&mode);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&N);
  free(frequency);
}



/*
Scene *SmoothObjInsideMask3(Scene *label, float r,
			    BMap *mask){
  AdjRel3 *A = Spheric(r);
  AdjVxl  *N;
  int fx,fy,fz,dp;
  int p,q,i,Lmax,l,lmax;
  int *frequency;
  Scene *flabel=NULL;

  flabel = CopyScene(label);
  Lmax = MaximumValue3(label);
  frequency = (int *)calloc(Lmax+1,sizeof(int));

  N = AdjVoxels(label, A);
  FrameSizes3(A, &fx, &fy, &fz);
  dp  = fx*1;
  dp += fy*label->xsize;
  dp += fz*label->xsize*label->ysize;

  for(p=dp; p<label->n-dp; p++){
    if(label->data[p]==0 ||
       !_fast_BMapGet(mask,p)) continue;

    memset(frequency, 0, (Lmax+1)*sizeof(int));

    for(i=0; i<N->n; i++){
      q = p + N->dp[i];
      frequency[label->data[q]]++;
    }
    lmax = label->data[p];
    for(l=0; l<=Lmax; l++){
      if(frequency[l]>frequency[lmax])
	lmax = l;
    }
    flabel->data[p] = lmax;
  }
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&N);
  free(frequency);

  return flabel;
}
*/
