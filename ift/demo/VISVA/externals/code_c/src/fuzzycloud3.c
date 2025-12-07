
#include "fuzzycloud3.h"


RegionCloud3 *LabelList2RegionCloud3(FileList *L){
  int nimages,i,l,Lmax,p,q;
  RegionCloud3 *rcloud=NULL;
  Scene *mask,*label;
  int Sx,Sy,Sz,s;     //sizes.
  Voxel **C = NULL;   //centroids.
  Voxel **MBBl = NULL;
  Voxel **MBBh = NULL;
  Voxel Vl,Vh,u,v,c;
  float dx=1.0,dy=1.0,dz=1.0;
  float sum_x,sum_y,sum_z,sum;

  //Find the maximum label:
  Lmax = 0;
  nimages = L->n;
  for(i=0; i<nimages; i++){
    label = ReadVolume( GetFile(L, i) );
    Lmax = MAX(Lmax, MaximumValue3(label));
    dx = label->dx;
    dy = label->dy;
    dz = label->dz;
    DestroyScene(&label);
  }

  //Initialize structures:
  C = (Voxel **) calloc(nimages, sizeof(Voxel *));
  MBBl = (Voxel **) calloc(nimages, sizeof(Voxel *));
  MBBh = (Voxel **) calloc(nimages, sizeof(Voxel *));
  if(C==NULL) Error(MSG1,"LabelList2RegionCloud3");
  for(i=0; i<nimages; i++){
    C[i] = (Voxel *) calloc(Lmax+1, sizeof(Voxel));
    if(C[i]==NULL) Error(MSG1,"LabelList2RegionCloud3");

    MBBl[i] = (Voxel *) calloc(Lmax+1, sizeof(Voxel));
    if(MBBl[i]==NULL) Error(MSG1,"LabelList2RegionCloud3");

    MBBh[i] = (Voxel *) calloc(Lmax+1, sizeof(Voxel));
    if(MBBh[i]==NULL) Error(MSG1,"LabelList2RegionCloud3");
  }

  //Compute centroids and MBBs:
  for(i=0; i<nimages; i++){
    label = ReadVolume( GetFile(L, i) );

    for(l=1; l<=Lmax; l++){
      mask = Threshold3(label, l, l);
      ComputeMaskCentroid3(mask, &c);
      ComputeSceneMBB(mask, &Vl, &Vh);
      C[i][l] = c;
      MBBl[i][l] = Vl;
      MBBh[i][l] = Vh;
      DestroyScene(&mask);
    }
    mask = Threshold3(label, 1, INT_MAX);
    ComputeMaskCentroid3(mask, &C[i][0]);
    DestroyScene(&mask);
    DestroyScene(&label);
  }

  //Create RegionCloud3:
  rcloud = (RegionCloud3 *)calloc(1,sizeof(RegionCloud3));
  if(rcloud==NULL) 
    Error(MSG1,"LabelList2RegionCloud3");
  rcloud->prob = (Scene **)calloc(Lmax+1,sizeof(Scene *));
  if(rcloud->prob==NULL) 
    Error(MSG1,"LabelList2RegionCloud3");
  rcloud->disp  = CreateAdjRel3(Lmax+1);
  rcloud->fdisp = CreateFloatAdjRel3(Lmax+1);
  rcloud->nobjs = Lmax;
  rcloud->nimages = nimages;

  for(l=1; l<=Lmax; l++){
    sum_x = sum_y = sum_z = 0.0;
    for(i=0; i<nimages; i++){
      sum_x += (C[i][l].x - C[i][0].x);
      sum_y += (C[i][l].y - C[i][0].y);
      sum_z += (C[i][l].z - C[i][0].z);
    }
    (rcloud->fdisp)->dx[l] = sum_x/(float)nimages;
    (rcloud->fdisp)->dy[l] = sum_y/(float)nimages;
    (rcloud->fdisp)->dz[l] = sum_z/(float)nimages;

    (rcloud->disp)->dx[l] = ROUND(sum_x/(float)nimages);
    (rcloud->disp)->dy[l] = ROUND(sum_y/(float)nimages);
    (rcloud->disp)->dz[l] = ROUND(sum_z/(float)nimages);

    Sx = Sy = Sz = 0;
    for(i=0; i<nimages; i++){
      c.x = C[i][l].x;
      c.y = C[i][l].y;
      c.z = C[i][l].z;
      Vl = MBBl[i][l];
      Vh = MBBh[i][l];

      s = MAX(c.x-Vl.x,Vh.x-c.x)*2+1;
      if(s > Sx) Sx = s;
      s = MAX(c.y-Vl.y,Vh.y-c.y)*2+1;
      if(s > Sy) Sy = s;
      s = MAX(c.z-Vl.z,Vh.z-c.z)*2+1;
      if(s > Sz) Sz = s;
    }
    rcloud->prob[l] = CreateScene(Sx, Sy, Sz);
    SetVoxelSize(rcloud->prob[l], dx, dy, dz);
  }

  //Compute Sum:
  for(i=0; i<nimages; i++){
    label = ReadVolume( GetFile(L, i) );
  
    for(p=0; p<label->n; p++){
      l = label->data[p];
      if(l==0) continue;
      u.x = VoxelX(label, p);
      u.y = VoxelY(label, p);
      u.z = VoxelZ(label, p);

      u.x -= C[i][l].x;
      u.y -= C[i][l].y;
      u.z -= C[i][l].z;

      v.x = u.x + (rcloud->prob[l])->xsize/2;
      v.y = u.y + (rcloud->prob[l])->ysize/2;
      v.z = u.z + (rcloud->prob[l])->zsize/2;

      if(ValidVoxel(rcloud->prob[l], v.x, v.y, v.z)){
	q = VoxelAddress(rcloud->prob[l],v.x,v.y,v.z);
	(rcloud->prob[l])->data[q] += 1;
      }
    }
    DestroyScene(&label);
  }

  //Sum to Probability:
  for(l=1; l<=Lmax; l++){
    for(p=0; p<(rcloud->prob[l])->n; p++){
      sum = (float)(rcloud->prob[l])->data[p];
      (rcloud->prob[l])->data[p] = ROUND(MAX_PROB*(sum/nimages));
    }
  }

  //Free memory:
  for(i=0; i<nimages; i++){
    free(C[i]);
    free(MBBl[i]);
    free(MBBh[i]);
  }
  free(C);
  free(MBBl);
  free(MBBh);

  return rcloud;
}



void DestroyRegionCloud3(RegionCloud3 **rcloud){
  RegionCloud3 *aux;
  int i;

  aux = *rcloud;
  if(aux != NULL){
    if(aux->disp!=NULL)
      DestroyAdjRel3(&aux->disp);
    if(aux->fdisp!=NULL)
      DestroyFloatAdjRel3(&aux->fdisp);

    if(aux->prob!=NULL){
      for(i=1; i<=aux->nobjs; i++)
	if(aux->prob[i]!=NULL)
	  DestroyScene(&aux->prob[i]);
      free(aux->prob);
    }
    free(aux);
    *rcloud = NULL;
  }
}


void          GetVoxelSizeRegionCloud3(RegionCloud3 *rcloud,
				       float *dx, 
				       float *dy, 
				       float *dz){
  *dx = (rcloud->prob[1])->dx;
  *dy = (rcloud->prob[1])->dy;
  *dz = (rcloud->prob[1])->dz;
}


RegionCloud3 *SubsamplingRegionCloud3(RegionCloud3 *rcloud){
  RegionCloud3 *sub=NULL;
  Scene *tmp=NULL,*blur=NULL;
  int nobjs,l;

  nobjs = rcloud->nobjs;

  //Create RegionCloud3:
  sub = (RegionCloud3 *)calloc(1,sizeof(RegionCloud3));
  if(sub==NULL) 
    Error(MSG1,"SubsamplingRegionCloud3");
  sub->prob = (Scene **)calloc(nobjs+1,sizeof(Scene *));
  if(sub->prob==NULL) 
    Error(MSG1,"SubsamplingRegionCloud3");
  sub->disp  = CreateAdjRel3(nobjs+1);
  sub->fdisp = CreateFloatAdjRel3(nobjs+1);
  sub->nobjs = rcloud->nobjs;
  sub->nimages = rcloud->nimages;

  for(l=1; l<=nobjs; l++){
    (sub->fdisp)->dx[l] = (rcloud->fdisp)->dx[l]/2.;
    (sub->fdisp)->dy[l] = (rcloud->fdisp)->dy[l]/2.;
    (sub->fdisp)->dz[l] = (rcloud->fdisp)->dz[l]/2.;

    (sub->disp)->dx[l] = ROUND((sub->fdisp)->dx[l]);
    (sub->disp)->dy[l] = ROUND((sub->fdisp)->dy[l]);
    (sub->disp)->dz[l] = ROUND((sub->fdisp)->dz[l]);

    tmp  = AddFrame3(rcloud->prob[l], 1, 0);
    blur = FastGaussianBlur3(tmp);
    sub->prob[l] = Subsampling3(blur);
    DestroyScene(&blur);
    DestroyScene(&tmp);
  }
  return sub;
}


RegionCloud3 *LinearInterpRegionCloud3(RegionCloud3 *rcloud,
				       float dx,float dy,float dz){
  RegionCloud3 *interp=NULL;
  float odx,ody,odz;
  int nobjs,l;

  nobjs = rcloud->nobjs;

  //Create RegionCloud3:
  interp = (RegionCloud3 *)calloc(1,sizeof(RegionCloud3));
  if(interp==NULL) 
    Error(MSG1,"LinearInterpRegionCloud3");
  interp->prob = (Scene **)calloc(nobjs+1,sizeof(Scene *));
  if(interp->prob==NULL) 
    Error(MSG1,"LinearInterpRegionCloud3");
  interp->disp  = CreateAdjRel3(nobjs+1);
  interp->fdisp = CreateFloatAdjRel3(nobjs+1);
  interp->nobjs = rcloud->nobjs;
  interp->nimages = rcloud->nimages;

  for(l=1; l<=nobjs; l++){
    odx = rcloud->prob[l]->dx;
    ody = rcloud->prob[l]->dy;
    odz = rcloud->prob[l]->dz;
    (interp->fdisp)->dx[l] = (rcloud->fdisp)->dx[l]*odx/dx;
    (interp->fdisp)->dy[l] = (rcloud->fdisp)->dy[l]*ody/dy;
    (interp->fdisp)->dz[l] = (rcloud->fdisp)->dz[l]*odz/dz;

    (interp->disp)->dx[l] = ROUND((interp->fdisp)->dx[l]);
    (interp->disp)->dy[l] = ROUND((interp->fdisp)->dy[l]);
    (interp->disp)->dz[l] = ROUND((interp->fdisp)->dz[l]);

    interp->prob[l] = FastLinearInterpCentr3(rcloud->prob[l], 
					     dx,dy,dz);
  }
  return interp;
}


RegionCloud3 *GaussianBlurRegionCloud3(RegionCloud3 *rcloud){
  RegionCloud3 *blur=NULL;
  Scene *tmp=NULL;
  int nobjs,l;

  nobjs = rcloud->nobjs;

  //Create RegionCloud3:
  blur = (RegionCloud3 *)calloc(1,sizeof(RegionCloud3));
  if(blur==NULL) 
    Error(MSG1,"GaussianBlurRegionCloud3");
  blur->prob = (Scene **)calloc(nobjs+1,sizeof(Scene *));
  if(blur->prob==NULL) 
    Error(MSG1,"GaussianBlurRegionCloud3");
  blur->disp  = CreateAdjRel3(nobjs+1);
  blur->fdisp = CreateFloatAdjRel3(nobjs+1);
  blur->nobjs = rcloud->nobjs;
  blur->nimages = rcloud->nimages;

  for(l=1; l<=nobjs; l++){
    (blur->fdisp)->dx[l] = (rcloud->fdisp)->dx[l];
    (blur->fdisp)->dy[l] = (rcloud->fdisp)->dy[l];
    (blur->fdisp)->dz[l] = (rcloud->fdisp)->dz[l];

    (blur->disp)->dx[l] = ROUND((blur->fdisp)->dx[l]);
    (blur->disp)->dy[l] = ROUND((blur->fdisp)->dy[l]);
    (blur->disp)->dz[l] = ROUND((blur->fdisp)->dz[l]);

    tmp = AddFrame3(rcloud->prob[l], 1, 0);
    blur->prob[l] = FastGaussianBlur3(tmp);
    DestroyScene(&tmp);
  }
  return blur;
}


RegionCloud3 *ChangeOrientationToLPSRegionCloud3(RegionCloud3 *rcloud,
						 char *ori){
  RegionCloud3 *lps=NULL;
  int nobjs,l;

  nobjs = rcloud->nobjs;

  //Create RegionCloud3:
  lps = (RegionCloud3 *)calloc(1,sizeof(RegionCloud3));
  if(lps==NULL) 
    Error(MSG1,"ChangeOrientationToLPSRegionCloud3");
  lps->prob = (Scene **)calloc(nobjs+1,sizeof(Scene *));
  if(lps->prob==NULL) 
    Error(MSG1,"ChangeOrientationToLPSRegionCloud3");
  lps->disp  = CreateAdjRel3(nobjs+1);
  lps->fdisp = ChangeOrientationToLPSFloatAdjRel3(rcloud->fdisp, ori);
  lps->nobjs = rcloud->nobjs;
  lps->nimages = rcloud->nimages;

  for(l=1; l<=nobjs; l++){
    (lps->disp)->dx[l] = ROUND((lps->fdisp)->dx[l]);
    (lps->disp)->dy[l] = ROUND((lps->fdisp)->dy[l]);
    (lps->disp)->dz[l] = ROUND((lps->fdisp)->dz[l]);

    lps->prob[l] = ChangeOrientationToLPS(rcloud->prob[l], ori);
  }
  return lps;
}



RegionCloud3 *ReadRegionCloud3(char *filename){
  RegionCloud3 *rcloud=NULL;
  char tmp[512],more[512];
  int nimages,nobjs,l;
  FILE *fp;

  fp = fopen(filename,"rb");
  if(fp == NULL)
    Error(MSG2,"ReadRegionCloud3");

  fread(&nimages,sizeof(int),1,fp);
  fread(&nobjs,  sizeof(int),1,fp);

  //Create RegionCloud3:
  rcloud = (RegionCloud3 *)calloc(1,sizeof(RegionCloud3));
  if(rcloud==NULL)
    Error(MSG1,"ReadRegionCloud3");
  rcloud->prob = (Scene **)calloc(nobjs+1,sizeof(Scene *));
  if(rcloud->prob==NULL)
    Error(MSG1,"ReadRegionCloud3");
  rcloud->disp  = CreateAdjRel3(nobjs+1);
  rcloud->fdisp = CreateFloatAdjRel3(nobjs+1);
  rcloud->nobjs   = nobjs;
  rcloud->nimages = nimages;

  fread((rcloud->fdisp)->dx,sizeof(float),nobjs+1,fp);
  fread((rcloud->fdisp)->dy,sizeof(float),nobjs+1,fp);
  fread((rcloud->fdisp)->dz,sizeof(float),nobjs+1,fp);
  fclose(fp);

  for(l=1; l<=nobjs; l++){
    (rcloud->disp)->dx[l] = ROUND((rcloud->fdisp)->dx[l]);
    (rcloud->disp)->dy[l] = ROUND((rcloud->fdisp)->dy[l]);
    (rcloud->disp)->dz[l] = ROUND((rcloud->fdisp)->dz[l]);

    strcpy(tmp, filename);
    RemoveFileExtension(tmp);
    sprintf(more, "_%02d.scn.bz2", l);
    strcat(tmp, more);
    rcloud->prob[l] = ReadScene(tmp);
  }

  return rcloud;
}


void WriteRegionCloud3(RegionCloud3 *rcloud, 
		       char *filename){
  char tmp[512],more[512];
  int l;
  FILE *fp;

  fp = fopen(filename,"wb");
  if(fp == NULL)
    Error(MSG2,"WriteRegionCloud3");

  fwrite(&rcloud->nimages,sizeof(int),1,fp);
  fwrite(&rcloud->nobjs,  sizeof(int),1,fp);
  fwrite((rcloud->fdisp)->dx,sizeof(float),rcloud->nobjs+1,fp);
  fwrite((rcloud->fdisp)->dy,sizeof(float),rcloud->nobjs+1,fp);
  fwrite((rcloud->fdisp)->dz,sizeof(float),rcloud->nobjs+1,fp);
  fclose(fp);

  for(l=1; l<=rcloud->nobjs; l++){
    strcpy(tmp, filename);
    RemoveFileExtension(tmp);
    sprintf(more, "_%02d.scn.bz2", l);
    strcat(tmp, more);
    WriteScene(rcloud->prob[l], tmp);
  }
}


void RemoveElemRegionCloud3(RegionCloud3 *rcloud,
			    Scene *label){
  Scene *mask=NULL;
  Voxel u,v;
  int p,q,l;
  float prob,sum;
  Voxel *C = NULL;   //centroids.
  
  C = (Voxel *) calloc(rcloud->nobjs+1, sizeof(Voxel));
  if(C==NULL) Error(MSG1,"RemoveElemRegionCloud3");

  //Compute centroids:
  for(l=1; l<=rcloud->nobjs; l++){
    mask = Threshold3(label, l, l);
    ComputeMaskCentroid3(mask, &C[l]);
    DestroyScene(&mask);
  }

  //Probability to Sum:
  for(l=1; l<=rcloud->nobjs; l++){
    for(p=0; p<(rcloud->prob[l])->n; p++){
      prob  = (float)(rcloud->prob[l])->data[p];
      prob /= (float)MAX_PROB;
      (rcloud->prob[l])->data[p] = ROUND(prob*rcloud->nimages);
    }
  }

  //Remove label:
  for(p=0; p<label->n; p++){
    l = label->data[p];
    if(l==0) continue;
    u.x = VoxelX(label, p);
    u.y = VoxelY(label, p);
    u.z = VoxelZ(label, p);

    u.x -= C[l].x;
    u.y -= C[l].y;
    u.z -= C[l].z;
    
    v.x = u.x + (rcloud->prob[l])->xsize/2;
    v.y = u.y + (rcloud->prob[l])->ysize/2;
    v.z = u.z + (rcloud->prob[l])->zsize/2;

    if(ValidVoxel(rcloud->prob[l], v.x,v.y,v.z)){
      q = VoxelAddress(rcloud->prob[l],v.x,v.y,v.z);
      (rcloud->prob[l])->data[q] -= 1;
    }
  }

  //Sum to Probability:
  rcloud->nimages--;
  for(l=1; l<=rcloud->nobjs; l++){
    for(p=0; p<(rcloud->prob[l])->n; p++){
      sum = (float)(rcloud->prob[l])->data[p];
      (rcloud->prob[l])->data[p] = ROUND(MAX_PROB*(sum/rcloud->nimages));
    }
  }
  free(C);
}


//-----------------------------------------

BorderCloud3 *RegionCloud2BorderCloud3(RegionCloud3 *rcloud){
  BorderCloud3 *bcloud=NULL;
  Scene *prob=NULL;
  ScnGradient *grad=NULL;
  float dx,dy,dz,r;
  int rv,Imax,l,nobjs = rcloud->nobjs;

  //Create BorderCloud3:
  bcloud = (BorderCloud3 *)calloc(1,sizeof(BorderCloud3));
  if(bcloud==NULL)
    Error(MSG1,"RegionCloud2BorderCloud3");
  bcloud->prob = (ScnGradient **)calloc(nobjs+1,sizeof(ScnGradient *));
  if(bcloud->prob==NULL)
    Error(MSG1,"RegionCloud2BorderCloud3");
  bcloud->disp  = CreateAdjRel3(nobjs+1);
  bcloud->fdisp = CreateFloatAdjRel3(nobjs+1);
  bcloud->nobjs = nobjs;

  for(l=1; l<=rcloud->nobjs; l++){
    dx = (rcloud->prob[l])->dx;
    dy = (rcloud->prob[l])->dy;
    dz = (rcloud->prob[l])->dz;

    //----------------------------------------
    r = 3.0;
    rv = MAX(ROUND(r/dx), 1);
    rv = MAX(ROUND(r/dy), rv);
    rv = MAX(ROUND(r/dz), rv);
    prob = AddFrame3(rcloud->prob[l], rv*2, 0);
    grad = SphericalScnGradient(prob, r);
    Imax = ScnGradientMaximumMag(grad);
    ScnGradientNormalize(grad, 0, Imax, 0, MAX_PROB);
    bcloud->prob[l] = RemScnGradientFrame(grad, rv);
    //PowerEnhancementScnGradient(bcloud->prob[l]);
    //----------------------------------------

    DestroyScene(&prob);
    DestroyScnGradient(&grad);
    (bcloud->disp)->dx[l] = (rcloud->disp)->dx[l];
    (bcloud->disp)->dy[l] = (rcloud->disp)->dy[l];
    (bcloud->disp)->dz[l] = (rcloud->disp)->dz[l];

    (bcloud->fdisp)->dx[l] = (rcloud->fdisp)->dx[l];
    (bcloud->fdisp)->dy[l] = (rcloud->fdisp)->dy[l];
    (bcloud->fdisp)->dz[l] = (rcloud->fdisp)->dz[l];
  }
  return bcloud;
}


void DestroyBorderCloud3(BorderCloud3 **bcloud){
  BorderCloud3 *aux;
  int i;

  aux = *bcloud;
  if(aux != NULL){
    if(aux->disp!=NULL)
      DestroyAdjRel3(&aux->disp);
    if(aux->fdisp!=NULL)
      DestroyFloatAdjRel3(&aux->fdisp);
    
    if(aux->prob!=NULL){
      for(i=1; i<=aux->nobjs; i++)
	if(aux->prob[i]!=NULL)
	  DestroyScnGradient(&aux->prob[i]);
      free(aux->prob);
    }
    free(aux);
    *bcloud = NULL;
  }
}


void          GetVoxelSizeBorderCloud3(BorderCloud3 *bcloud,
				       float *dx, 
				       float *dy, 
				       float *dz){
  *dx = ((bcloud->prob[1])->Gx)->dx;
  *dy = ((bcloud->prob[1])->Gx)->dy;
  *dz = ((bcloud->prob[1])->Gx)->dz;
}



BorderCloud3 *SubsamplingBorderCloud3(BorderCloud3 *bcloud){
  BorderCloud3 *sub=NULL;
  Scene *tmp=NULL;
  int l;

  //Create BorderCloud3:
  sub = (BorderCloud3 *)calloc(1,sizeof(BorderCloud3));
  if(sub==NULL)
    Error(MSG1,"SubsamplingBorderCloud3");
  sub->prob = (ScnGradient **)calloc(bcloud->nobjs+1,sizeof(ScnGradient *));
  if(sub->prob==NULL) 
    Error(MSG1,"SubsamplingBorderCloud3");
  sub->disp  = CreateAdjRel3(bcloud->nobjs+1);
  sub->fdisp = CreateFloatAdjRel3(bcloud->nobjs+1);
  sub->nobjs = bcloud->nobjs;

  for(l=1; l<=bcloud->nobjs; l++){
    (sub->fdisp)->dx[l] = (bcloud->fdisp)->dx[l]/2.;
    (sub->fdisp)->dy[l] = (bcloud->fdisp)->dy[l]/2.;
    (sub->fdisp)->dz[l] = (bcloud->fdisp)->dz[l]/2.;

    (sub->disp)->dx[l] = ROUND((sub->fdisp)->dx[l]);
    (sub->disp)->dy[l] = ROUND((sub->fdisp)->dy[l]);
    (sub->disp)->dz[l] = ROUND((sub->fdisp)->dz[l]);

    sub->prob[l] = (ScnGradient *)calloc(1,sizeof(ScnGradient));
    if(sub->prob[l] == NULL)
      Error(MSG1,"SubsamplingBorderCloud3");
    (sub->prob[l])->mag = NULL;

    tmp  = FastGaussianBlur3((bcloud->prob[l])->Gx);
    (sub->prob[l])->Gx = Subsampling3(tmp);
    DestroyScene(&tmp);

    tmp  = FastGaussianBlur3((bcloud->prob[l])->Gy);
    (sub->prob[l])->Gy = Subsampling3(tmp);
    DestroyScene(&tmp);

    tmp  = FastGaussianBlur3((bcloud->prob[l])->Gz);
    (sub->prob[l])->Gz = Subsampling3(tmp);
    DestroyScene(&tmp);
  }
  return sub;
}


BorderCloud3 *LinearInterpBorderCloud3(BorderCloud3 *bcloud,
				       float dx,float dy,float dz){
  BorderCloud3 *interp=NULL;
  float odx,ody,odz;
  int nobjs,l;

  nobjs = bcloud->nobjs;

  //Create BorderCloud3:
  interp = (BorderCloud3 *)calloc(1,sizeof(BorderCloud3));
  if(interp==NULL)
    Error(MSG1,"LinearInterpBorderCloud3");
  interp->prob = (ScnGradient **)calloc(nobjs+1,sizeof(ScnGradient *));
  if(interp->prob==NULL) 
    Error(MSG1,"LinearInterpBorderCloud3");
  interp->disp  = CreateAdjRel3(nobjs+1);
  interp->fdisp = CreateFloatAdjRel3(nobjs+1);
  interp->nobjs = bcloud->nobjs;

  for(l=1; l<=nobjs; l++){
    odx = ((bcloud->prob[l])->Gx)->dx;
    ody = ((bcloud->prob[l])->Gx)->dy;
    odz = ((bcloud->prob[l])->Gx)->dz;
    (interp->fdisp)->dx[l] = (bcloud->fdisp)->dx[l]*odx/dx;
    (interp->fdisp)->dy[l] = (bcloud->fdisp)->dy[l]*ody/dy;
    (interp->fdisp)->dz[l] = (bcloud->fdisp)->dz[l]*odz/dz;

    (interp->disp)->dx[l] = ROUND((interp->fdisp)->dx[l]);
    (interp->disp)->dy[l] = ROUND((interp->fdisp)->dy[l]);
    (interp->disp)->dz[l] = ROUND((interp->fdisp)->dz[l]);

    interp->prob[l] = LinearInterpScnGradientCentr(bcloud->prob[l], 
						   dx,dy,dz);
  }
  return interp;
}


BorderCloud3 *ChangeOrientationToLPSBorderCloud3(BorderCloud3 *bcloud,
						 char *ori){
  BorderCloud3 *lps=NULL;
  int nobjs,l;

  nobjs = bcloud->nobjs;

  //Create BorderCloud3:
  lps = (BorderCloud3 *)calloc(1,sizeof(BorderCloud3));
  if(lps==NULL) 
    Error(MSG1,"ChangeOrientationToLPSBorderCloud3");
  lps->prob = (ScnGradient **)calloc(nobjs+1,sizeof(ScnGradient *));
  if(lps->prob==NULL) 
    Error(MSG1,"ChangeOrientationToLPSBorderCloud3");
  lps->disp  = CreateAdjRel3(nobjs+1);
  lps->fdisp = ChangeOrientationToLPSFloatAdjRel3(bcloud->fdisp, ori);
  lps->nobjs = bcloud->nobjs;

  for(l=1; l<=nobjs; l++){
    (lps->disp)->dx[l] = ROUND((lps->fdisp)->dx[l]);
    (lps->disp)->dy[l] = ROUND((lps->fdisp)->dy[l]);
    (lps->disp)->dz[l] = ROUND((lps->fdisp)->dz[l]);

    lps->prob[l] = ChangeOrientationToLPS_ScnGradient(bcloud->prob[l], ori);
  }
  return lps;
}


void          NormalizeBorderCloud3(BorderCloud3 *bcloud,
				    int omin,int omax,
				    int nmin,int nmax){
  int nobjs,l;

  nobjs = bcloud->nobjs;
  for(l=1; l<=nobjs; l++){
    ScnGradientNormalize(bcloud->prob[l],
			 omin,omax,
			 nmin,nmax);
  }
}


BorderCloud3 *ReadBorderCloud3(char *filename){
  BorderCloud3 *bcloud=NULL;
  char tmp[512],more[512];
  int nobjs,l;
  FILE *fp;

  fp = fopen(filename,"rb");
  if(fp == NULL)
    Error(MSG2,"ReadBorderCloud3");

  fread(&nobjs,  sizeof(int),1,fp);

  //Create BorderCloud3:
  bcloud = (BorderCloud3 *)calloc(1,sizeof(BorderCloud3));
  if(bcloud==NULL)
    Error(MSG1,"ReadBorderCloud3");
  bcloud->prob = (ScnGradient **)calloc(nobjs+1,sizeof(ScnGradient *));
  if(bcloud->prob==NULL)
    Error(MSG1,"ReadBorderCloud3");
  bcloud->disp  = CreateAdjRel3(nobjs+1);
  bcloud->fdisp = CreateFloatAdjRel3(nobjs+1);
  bcloud->nobjs = nobjs;

  fread((bcloud->fdisp)->dx,sizeof(float),nobjs+1,fp);
  fread((bcloud->fdisp)->dy,sizeof(float),nobjs+1,fp);
  fread((bcloud->fdisp)->dz,sizeof(float),nobjs+1,fp);
  fclose(fp);

  for(l=1; l<=nobjs; l++){
    (bcloud->disp)->dx[l] = ROUND((bcloud->fdisp)->dx[l]);
    (bcloud->disp)->dy[l] = ROUND((bcloud->fdisp)->dy[l]);
    (bcloud->disp)->dz[l] = ROUND((bcloud->fdisp)->dz[l]);

    strcpy(tmp, filename);
    RemoveFileExtension(tmp);
    sprintf(more, "_%02d.sgr.bz2", l);
    strcat(tmp, more);
    bcloud->prob[l] = ReadCompressedScnGradient(tmp);
  }

  return bcloud;
}


void WriteBorderCloud3(BorderCloud3 *bcloud, 
		       char *filename){
  char tmp[512],more[512];
  int l;
  FILE *fp;

  fp = fopen(filename,"wb");
  if(fp == NULL)
    Error(MSG2,"WriteBorderCloud3");

  fwrite(&bcloud->nobjs,  sizeof(int),1,fp);
  fwrite((bcloud->fdisp)->dx,sizeof(float),bcloud->nobjs+1,fp);
  fwrite((bcloud->fdisp)->dy,sizeof(float),bcloud->nobjs+1,fp);
  fwrite((bcloud->fdisp)->dz,sizeof(float),bcloud->nobjs+1,fp);
  fclose(fp);

  for(l=1; l<=bcloud->nobjs; l++){
    strcpy(tmp, filename);
    RemoveFileExtension(tmp);
    sprintf(more, "_%02d.sgr.bz2", l);
    strcat(tmp, more);
    WriteCompressedScnGradient(bcloud->prob[l], tmp);
  }
}


