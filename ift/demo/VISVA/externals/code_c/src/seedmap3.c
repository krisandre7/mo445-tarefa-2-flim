
#include "seedmap3.h"


AdjSeedmap3 *CreateAdjSeedmap3(int nobjs){
  AdjSeedmap3 *asmap=NULL;
  int l;

  asmap = (AdjSeedmap3 *) calloc(1,sizeof(AdjSeedmap3));
  if(asmap == NULL)
    Error(MSG1,"CreateAdjSeedmap3");

  asmap->disp = CreateAdjRel3(nobjs+1);
  asmap->object = (Cloud3 **) calloc(nobjs+1,sizeof(Cloud3 *));
  asmap->obj_border = (Cloud3 **) calloc(nobjs+1,sizeof(Cloud3 *));
  asmap->bkg_border = (Cloud3 **) calloc(nobjs+1,sizeof(Cloud3 *));
  asmap->uncertainty = (Cloud3 **) calloc(nobjs+1,sizeof(Cloud3 *));

  if(asmap->object==NULL || 
     asmap->obj_border==NULL ||
     asmap->bkg_border==NULL ||
     asmap->uncertainty==NULL)
    Error(MSG1,"CreateAdjSeedmap3");  

  for(l=0; l<=nobjs; l++){
    asmap->object[l] = NULL;
    asmap->obj_border[l] = NULL;
    asmap->bkg_border[l] = NULL;
    asmap->uncertainty[l] = NULL;
  }
  asmap->nobjs = nobjs;

  return asmap;
}


void DestroyAdjSeedmap3(AdjSeedmap3 **asmap){
  AdjSeedmap3 *aux;
  int i;

  aux = *asmap;
  if(aux != NULL){
    if(aux->disp!=NULL)
      DestroyAdjRel3(&aux->disp);

    if(aux->uncertainty!=NULL){
      for(i=0; i<=aux->nobjs; i++)
	if(aux->uncertainty[i]!=NULL)
	  DestroyCloud3(&aux->uncertainty[i]);
      free(aux->uncertainty);
    }
    if(aux->object!=NULL){
      for(i=0; i<=aux->nobjs; i++)
	if(aux->object[i]!=NULL)
	  DestroyCloud3(&aux->object[i]);
      free(aux->object);
    }
    if(aux->obj_border!=NULL){
      for(i=0; i<=aux->nobjs; i++)
	if(aux->obj_border[i]!=NULL)
	  DestroyCloud3(&aux->obj_border[i]);
      free(aux->obj_border);
    }
    if(aux->bkg_border!=NULL){
      for(i=0; i<=aux->nobjs; i++)
	if(aux->bkg_border[i]!=NULL)
	  DestroyCloud3(&aux->bkg_border[i]);
      free(aux->bkg_border);
    }
    free(aux);
    *asmap = NULL;
  }
}


AdjSeedmap3 *RegionCloud2AdjSeedmap3(RegionCloud3 *rcloud){
  AdjSeedmap3 *asmap=NULL;
  Scene *mask=NULL,*fmask=NULL;
  Scene *bmask=NULL,*fbmask=NULL;
  Scene *prob=NULL;
  int nimgs,nobjs,l;
  AdjRel3 *A=NULL;
  Voxel Ref;

  A = Spheric(1.0);
  nobjs = rcloud->nobjs;
  nimgs = rcloud->nimages;
  asmap = CreateAdjSeedmap3(nobjs);

  for(l=1; l<=nobjs; l++){
    prob = rcloud->prob[l];

    (asmap->disp)->dx[l] = (rcloud->disp)->dx[l];
    (asmap->disp)->dy[l] = (rcloud->disp)->dy[l];
    (asmap->disp)->dz[l] = (rcloud->disp)->dz[l];

    mask = Threshold3(prob, MAX_PROB, MAX_PROB);
    Ref.x = mask->xsize/2;
    Ref.y = mask->ysize/2;
    Ref.z = mask->zsize/2;
    asmap->object[l] = Mask2Cloud3(mask, Ref);

    bmask = GetBorder3(mask, A);
    asmap->obj_border[l] = Mask2Cloud3(bmask, Ref);
    DestroyScene(&bmask);
    DestroyScene(&mask);

    mask = Threshold3(prob, 1, MAX_PROB-1);
    asmap->uncertainty[l] = Mask2Cloud3(mask, Ref);
    DestroyScene(&mask);

    mask  = Threshold3(prob, 0, 0);
    fmask = AddFrame3(mask, 2, 1);
    fbmask = GetBorder3(fmask, A);
    bmask = RemFrame3(fbmask, 1);
    Ref.x += 1;
    Ref.y += 1;
    Ref.z += 1;
    asmap->bkg_border[l] = Mask2Cloud3(bmask, Ref);
    DestroyScene(&fbmask);
    DestroyScene(&bmask);
    DestroyScene(&fmask);
    DestroyScene(&mask);
  }
  DestroyAdjRel3(&A);

  return asmap;
}



void DrawAdjSeedmapObject3(Scene *scn,
			   AdjSeedmap3 *asmap,
			   Voxel u,
			   int l,
			   int val){
  Voxel du;
  du.x = u.x + (asmap->disp)->dx[l];
  du.y = u.y + (asmap->disp)->dy[l];
  du.z = u.z + (asmap->disp)->dz[l];
  DrawCloud32Scene(asmap->object[l],
		   scn, du, val);
}


void DrawAdjSeedmapObjBorder3(Scene *scn,
			      AdjSeedmap3 *asmap,
			      Voxel u,
			      int l,
			      int val){
  Voxel du;
  du.x = u.x + (asmap->disp)->dx[l];
  du.y = u.y + (asmap->disp)->dy[l];
  du.z = u.z + (asmap->disp)->dz[l];
  DrawCloud32Scene(asmap->obj_border[l],
		   scn, du, val);
}


void DrawAdjSeedmapBkgBorder3(Scene *scn,
			      AdjSeedmap3 *asmap,
			      Voxel u,
			      int l,
			      int val){
  Voxel du;
  du.x = u.x + (asmap->disp)->dx[l];
  du.y = u.y + (asmap->disp)->dy[l];
  du.z = u.z + (asmap->disp)->dz[l];
  DrawCloud32Scene(asmap->bkg_border[l],
		   scn, du, val);
}


void DrawAdjSeedmapUncertainty3(Scene *scn,
				AdjSeedmap3 *asmap,
				Voxel u,
				int l,
				int val){
  Voxel du;
  int p;
  du.x = u.x + (asmap->disp)->dx[l];
  du.y = u.y + (asmap->disp)->dy[l];
  du.z = u.z + (asmap->disp)->dz[l];

  if(!Cloud3FitInside(asmap->bkg_border[l],
		      du, scn, 1)){
    DrawCloud32Scene(asmap->uncertainty[l],
		     scn, du, val);
  }
  else{
    p = VoxelAddress(scn,du.x,du.y,du.z);
    DrawOptCloud32Scene(asmap->uncertainty[l],
			scn, p, val);
  }
}


void CopyAdjSeedmapUncertainty3(Scene *dest,
				Scene *src,
				AdjSeedmap3 *asmap,
				Voxel u,
				int l){
  Cloud3 *S=NULL;
  Voxel du,v;
  int i,p,q;
  du.x = u.x + (asmap->disp)->dx[l];
  du.y = u.y + (asmap->disp)->dy[l];
  du.z = u.z + (asmap->disp)->dz[l];  

  if(!Cloud3FitInside(asmap->bkg_border[l],
		      du, dest, 1)){
    S = asmap->uncertainty[l];
    for(i=0; i<S->n; i++){
      v.x = du.x + S->dx[i];
      v.y = du.y + S->dy[i];
      v.z = du.z + S->dz[i];
      if(ValidVoxel(dest,v.x,v.y,v.z)){
	p = VoxelAddress(dest,v.x,v.y,v.z);
	dest->data[p] = src->data[p];
      }
    }
  }
  else{ //Cloud3FitInside
    p = VoxelAddress(dest,du.x,du.y,du.z);
    S = asmap->uncertainty[l];
    OptimizeCloud3(S, dest);
    for(i=0; i<S->n; i++){
      q = p + S->dp[i];
      dest->data[q] = src->data[q];
    }
  }
}


void AddAdjSeedmapUncertainty3(Scene *dest,
			       Scene *src,
			       AdjSeedmap3 *asmap,
			       Voxel u,
			       int l){
  Cloud3 *S=NULL;
  Voxel du,v;
  int i,p,q;
  du.x = u.x + (asmap->disp)->dx[l];
  du.y = u.y + (asmap->disp)->dy[l];
  du.z = u.z + (asmap->disp)->dz[l];  

  if(!Cloud3FitInside(asmap->bkg_border[l],
		      du, dest, 1)){
    S = asmap->uncertainty[l];
    for(i=0; i<S->n; i++){
      v.x = du.x + S->dx[i];
      v.y = du.y + S->dy[i];
      v.z = du.z + S->dz[i];
      if(ValidVoxel(dest,v.x,v.y,v.z)){
	p = VoxelAddress(dest,v.x,v.y,v.z);
	dest->data[p] += src->data[p];
      }
    }
  }
  else{ //Cloud3FitInside
    p = VoxelAddress(dest,du.x,du.y,du.z);
    S = asmap->uncertainty[l];
    OptimizeCloud3(S, dest);
    for(i=0; i<S->n; i++){
      q = p + S->dp[i];
      dest->data[q] += src->data[q];
    }
  }
}


/*
void CloudArcWeight3(Scene *arcw, 
		     Scene *grad,
		     Voxel u,
		     BorderCloud3 *bcloud,
		     Seedmap3 *smap,
		     AdjSeedmap3 *asmap,
		     int l,
		     float w){
  Cloud3 *S=NULL;
  Scene *prob=NULL;
  Voxel v,b,du;
  int i,p,q,run;
  int dx,dy,dz;
  float v1,v2;

  du.x = u.x + (asmap->disp)->dx[l];
  du.y = u.y + (asmap->disp)->dy[l];
  du.z = u.z + (asmap->disp)->dz[l];
  for(run=1; run<=3; run++){
    if(run==1) S = asmap->uncertainty[l];
    if(run==2) S = asmap->obj_border[l];
    if(run==3) S = asmap->bkg_border[l];

    for(i=0; i<S->n; i++){
      dx = S->dx[i];
      dy = S->dy[i];
      dz = S->dz[i];

      v.x = du.x + dx;
      v.y = du.y + dy;
      v.z = du.z + dz;

      if(ValidVoxel(arcw,v.x,v.y,v.z)){
	p = VoxelAddress(arcw,v.x,v.y,v.z);

	v2 = (float)grad->data[p];
	
	prob = bcloud->prob[l];
	b.x = prob->xsize/2 + dx;
	b.y = prob->ysize/2 + dy;
	b.z = prob->zsize/2 + dz;
	if(ValidVoxel(prob,b.x,b.y,b.z)){
	  q = VoxelAddress(prob,b.x,b.y,b.z);
	  
	  v1 = (float)prob->data[q];
	}
	else v1 = 0.0;

	arcw->data[p] = ROUND(w*v1+(1.0-w)*v2);
      }
    }
  }
}
*/


void CloudArcWeight3(Scene *arcw,
		     Scene *wobj,
		     ScnGradient *grad,
		     Voxel u,
		     BorderCloud3 *bcloud,
		     AdjSeedmap3 *asmap,
		     int l, float w){
  Cloud3 *S=NULL;
  ScnGradient *prob=NULL;
  Voxel v,b,du;
  int i,p,q,run;
  int dx,dy,dz;
  float v1,v2,dotproduct,G;//,cos,gr;

  prob = bcloud->prob[l];
  ComputeScnGradientMagnitude(prob);

  G = (((float)(prob->mag)->maxval)*
       ((float)(grad->mag)->maxval));
  //printf("prob: %d, grad: %d\n",(prob->mag)->maxval, (grad->mag)->maxval);
  //printf("G: %f\n",G);
  if(G==0.0) G = 0.01;
  du.x = u.x + (asmap->disp)->dx[l];
  du.y = u.y + (asmap->disp)->dy[l];
  du.z = u.z + (asmap->disp)->dz[l];
  for(run=1; run<=3; run++){
    if(run==1) S = asmap->uncertainty[l];
    if(run==2) S = asmap->obj_border[l];
    if(run==3) S = asmap->bkg_border[l];

    for(i=0; i<S->n; i++){
      dx = S->dx[i];
      dy = S->dy[i];
      dz = S->dz[i];

      v.x = du.x + dx;
      v.y = du.y + dy;
      v.z = du.z + dz;

      if(ValidVoxel(arcw,v.x,v.y,v.z)){
	p = VoxelAddress(arcw,v.x,v.y,v.z);

	v2 = (float)wobj->data[p];
	
	b.x = (prob->Gx)->xsize/2 + dx;
	b.y = (prob->Gx)->ysize/2 + dy;
	b.z = (prob->Gx)->zsize/2 + dz;
	if(ValidVoxel(prob->Gx,b.x,b.y,b.z)){
	  q = VoxelAddress(prob->Gx,b.x,b.y,b.z);
	  v1 = (float)(prob->mag)->data[q];

	  //-------------------
	  //PowerEnhancement:
	  v1 = (v1*v1)/((float)(prob->mag)->maxval);
	  //-------------------

	  dotproduct  = grad->Gx->data[p]*prob->Gx->data[q];
	  dotproduct += grad->Gy->data[p]*prob->Gy->data[q];
	  dotproduct += grad->Gz->data[p]*prob->Gz->data[q];
	  //gr = (float)(grad->mag)->data[p];
	  //if(v1*gr>0.0) cos = dotproduct/(v1*gr);
	  //else          cos = 0.0;
	}
	else{ 
	  v1  = 0.0;
	  dotproduct = 0.0;
	  //cos = 0.0;
	  //gr  = 0.0;
	}

	//if(cos>0.0) cos = 0.0;
	if(dotproduct>0.0) dotproduct = 0.0;
	arcw->data[p] = ROUND((w*v1+(1.0-w)*v2)/(1.0-2.*dotproduct/G));
      }
    }
  }
}



