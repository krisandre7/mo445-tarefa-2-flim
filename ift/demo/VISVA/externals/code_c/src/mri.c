
#include "mri.h"


MRI_Info EstimateMRI_Information(Scene *scn){
  MRI_Info info;

  info.Tup  = MRI_UpperCut(scn);
  info.COG  = MRI_CentreOfGravity(scn, info.Tup);
  info.Tcsf = MRI_EstimateCSF(scn, info.COG, &(info.Ecsf));
  MRI_EstimateGM_WM(scn, info.Tcsf, info.Ecsf, info.Tup,
		    &(info.Tgm), &(info.Twm));
  return info;
}


int MRI_UpperCut(Scene *scn){
  int Totsu;
  Curve *hist=NULL,*chist=NULL;
  double cmax;
  int i,T;

  hist  = Histogram3(scn);
  Totsu = OtsuHistogramThreshold(hist);

  chist = XClipHistogram(hist, Totsu, INT_MAX);
  DestroyCurve(&hist);
  cmax = CurveMaximum(chist);

  for(i=chist->n-1; i>Totsu; i--)
    if(chist->Y[i]>cmax*0.20) break;

  T = ROUND(i + (chist->n-1-i)*0.05);
  DestroyCurve(&chist);
  return T;
}

int MRI_LowerCut(Scene *scn){
  int Totsu;
  Curve *hist=NULL,*chist=NULL;
  double cmax;
  int i,T;

  hist  = Histogram3(scn);
  Totsu = OtsuHistogramThreshold(hist);

  chist = XClipHistogram(hist, Totsu, INT_MAX);
  DestroyCurve(&hist);
  cmax = CurveMaximum(chist);

  for(i=Totsu; i<chist->n; i++)
    if(chist->Y[i]>cmax*0.20) break;

  T = i;
  DestroyCurve(&chist);
  return T;
}


int MRI_LowerBoundCSF(Scene *scn){
  int T,T1;
  T = ComputeOtsu3(scn);
  T1 = SceneRangeMeanValue(scn, 0, T);
  T1 = MAX(1,(T1+2*T)/3);
  return T1;
}


Voxel  MRI_CentreOfGravity(Scene *scn, int Tup){
  Scene *mask=NULL;
  Voxel COG;
  int Tcsf;
  Tcsf = MRI_LowerBoundCSF(scn);
  mask = MRI_InternalSeeds(scn, Tcsf, 0, Tcsf, Tup);
  ComputeMaskCentroid3(mask, &COG);
  DestroyScene(&mask);
  return COG;
}

int    MRI_EstimateCSF(Scene *scn, Voxel COG, int *Ecsf){
  Scene *rscn=NULL;
  Curve *hist=NULL,*tmp=NULL;
  Voxel Vm,VM;
  int T,T1,T2;
  //The ventricles are filled with cerebrospinal fluid (CSF).
  Vm.x = COG.x - ROUND(25.0/scn->dx);
  Vm.y = COG.y - ROUND(25.0/scn->dy);
  Vm.z = COG.z - ROUND(25.0/scn->dz);

  VM.x = COG.x + ROUND(25.0/scn->dx);
  VM.y = COG.y + ROUND(25.0/scn->dy);
  VM.z = COG.z + ROUND(25.0/scn->dz);

  rscn = CopySubScene(scn, Vm, VM);

  //WriteScene(rscn,"rscn.scn");

  hist = Histogram3(rscn);
  T = OtsuHistogramThreshold(hist);
  tmp = XClipHistogram(hist, 0, T);
  T2 = OtsuHistogramThreshold(tmp);
  DestroyCurve(&tmp);
  tmp = XClipHistogram(hist, 0, T2);
  T1 = AreaPercentageLowerThreshold(tmp, 25);
  //T1 = HistogramMedianValue(tmp);
  DestroyScene(&rscn);
  DestroyCurve(&hist);
  DestroyCurve(&tmp);
  *Ecsf = MAX(abs(T2-T1),1);
  return T1;
}


void MRI_EstimateGM_WM(Scene *scn, int Tcsf, int Ecsf, int Tup,
		       int *Tgm, int *Twm){
  Scene *mask=NULL;
  Curve *hist=NULL,*tmp=NULL;
  int T;

  mask = MRI_InternalSeeds(scn, Tcsf, Ecsf, 
			   Tcsf+Ecsf, Tup);

  if(MaximumValue3(mask)==0){ 
    DestroyScene(&mask);
    *Tgm = Tcsf+2*Ecsf; 
    *Twm = Tup-Ecsf;
    return;
  }

  hist = HistogramMask3(scn, mask);
  T = OtsuHistogramThreshold(hist);
  tmp = XClipHistogram(hist, 0, T);
  *Tgm = HistogramMedianValue(tmp);
  DestroyCurve(&tmp);
  tmp = XClipHistogram(hist, T, INT_MAX);
  *Twm = HistogramMedianValue(tmp);
  DestroyCurve(&tmp);
  DestroyScene(&mask);
  DestroyCurve(&hist);
}


Scene *MRI_InternalSeeds(Scene *scn, int Tcsf, int Ecsf,
			 int Tgm, int Tup){
  Set *S=NULL;
  Scene *mask,*tmp;
  float r;

  //The average skull thickness is 7.1 mm. 
  //-> upper bound 8 mm.
  r = 8.0;

  tmp = Threshold3(scn, (Tcsf+Ecsf+Tgm)/2, Tup);

  //WriteScene(tmp,"threshold.scn");

  SetSceneFrame(tmp, 1, 0);
  mask = ErodeBinScn(tmp, &S, r/2);

  //WriteScene(mask,"erosion.scn");

  SelectLargestComp(mask);
  DestroyScene(&tmp);
  if(S!=NULL) DestroySet(&S);
  mask->dx = scn->dx;
  mask->dy = scn->dy;
  mask->dz = scn->dz;

  return mask;
}


void MRI_SuppressSkull(Scene *arcw, Scene *scn, MRI_Info info){
  Set *S=NULL;
  Scene *mask,*tmp;
  float r,r2 = 35.0*35.0;
  Voxel u;
  int p;
  //static int id=1;

  //The average skull thickness is 7.1 mm. 
  //-> upper bound 8 mm.
  r = 8.0;

  tmp = Threshold3(scn, info.Tcsf+info.Ecsf, info.Tup);
  SetSceneFrame(tmp, 1, 0);

  mask = ErodeBinScnRadial(tmp, info.COG, r/2);
  DestroyScene(&tmp);

  tmp  = mask;
  mask = DilateBinScn(tmp, &S, r/2+3.0);
  DestroyScene(&tmp);
  if(S!=NULL) DestroySet(&S);

  //if(id==1) WriteScene(mask, "debug_mask_1.scn");
  //if(id==2) WriteScene(mask, "debug_mask_2.scn");
  //if(id==3) WriteScene(mask, "debug_mask_3.scn");
  //id++;

  for(p=0; p<scn->n; p++){
    u.x = VoxelX(scn,p);
    u.y = VoxelY(scn,p);
    u.z = VoxelZ(scn,p);

    if(mask->data[p]==0 &&
       VoxelEuclideanSquaredDistance(scn, u,info.COG)>r2)
      arcw->data[p] /= 2;
  }
  DestroyScene(&mask);
}



void MRI_SuppressOpticNerve(Scene *arcw, Scene *scn, MRI_Info info){
  Scene *sub,*mask,*tmp,*fsub;
  BoxSelection3 box;
  Set *S=NULL;
  int p;
  static int id=1;

  //Assumes that the data is in LPS orientation.
  box.v1.x = 1; box.v2.x = scn->xsize-2;
  box.v1.y = 1; box.v2.y = info.COG.y - ROUND(20.0/scn->dy);
  box.v1.z = 1; box.v2.z = info.COG.z;
  mask = Threshold3(scn, info.Tcsf+info.Ecsf, info.Tup);
  sub = CopySubScene(mask, box.v1, box.v2);
  SetScene(mask, 1);

  PasteSubScene(mask, sub, box.v1);
  if(id==1) WriteScene(mask, "debug_orig_1.scn");
  if(id==2) WriteScene(mask, "debug_orig_2.scn");
  if(id==3) WriteScene(mask, "debug_orig_3.scn");

  tmp  = ErodeBinScn(sub,&S,3.0);

  PasteSubScene(mask, tmp, box.v1);
  if(id==1) WriteScene(mask, "debug_erode_1.scn");
  if(id==2) WriteScene(mask, "debug_erode_2.scn");
  if(id==3) WriteScene(mask, "debug_erode_3.scn");

  fsub = DilateBinScn(tmp,&S,5.0);

  PasteSubScene(mask, fsub, box.v1);
  if(id==1) WriteScene(mask, "debug_dilate_1.scn");
  if(id==2) WriteScene(mask, "debug_dilate_2.scn");
  if(id==3) WriteScene(mask, "debug_dilate_3.scn");

  if(S!=NULL) DestroySet(&S);
  DestroyScene(&tmp);

  SetSceneFrame(fsub, 1, 1);
  PasteSubScene(mask, fsub, box.v1);
  DestroyScene(&fsub);
  DestroyScene(&sub);

  if(id==1) WriteScene(mask, "debug_mask_1.scn");
  if(id==2) WriteScene(mask, "debug_mask_2.scn");
  if(id==3) WriteScene(mask, "debug_mask_3.scn");
  id++;

  for(p=0; p<scn->n; p++){
    if(mask->data[p]==0)
      arcw->data[p] /= 2;
  }
  DestroyScene(&mask);
}



void   MRI_SuppressNonBrainBorders(Scene *arcw,
				   Scene *scn,
				   ScnGradient *grad,
				   MRI_Info info){
  Vector v;
  Voxel a,b;
  int m,max;
  int p,q,d;
  float *weight=NULL;

  weight = SceneIntensity2Gaussian(scn, info.Twm,
				   (info.Tgm-info.Tcsf),
				   (info.Twm-info.Tgm)*2);

  ComputeScnGradientMagnitude(grad);
  for(p=0; p<arcw->n; p++){
    a.x = VoxelX(arcw,p);
    a.y = VoxelY(arcw,p);
    a.z = VoxelZ(arcw,p);

    m = grad->mag->data[p];
    if(m==0) continue;

    v.x = grad->Gx->data[p]/((float)m);
    v.y = grad->Gy->data[p]/((float)m);
    v.z = grad->Gz->data[p]/((float)m);

    max = scn->data[p];
    for(d=1; d<=3; d++){
      b.x = a.x + ROUND((v.x*d)/arcw->dx);
      b.y = a.y + ROUND((v.y*d)/arcw->dy);
      b.z = a.z + ROUND((v.z*d)/arcw->dz);
      if(ValidVoxel(arcw, b.x,b.y,b.z)){
	q = VoxelAddress(arcw, b.x,b.y,b.z);
	if(scn->data[q]>max) max = scn->data[q];
      }
      else{ break; }
    }

    if(max>info.Tgm)
      arcw->data[p] = ROUND(arcw->data[p]*(weight[max]/2.0 + 0.5));
  }
  free(weight);
}


void ComputeMRISceneMBB(Scene *scn, 
			Voxel *Vm, 
			Voxel *VM){
  Scene *bin=NULL;
  int T,T1,ysize,zsize;
  int dx1,dx2;

  T = ComputeOtsu3(scn);
  T1 = SceneRangeMeanValue(scn, 0, T);
  T1 = MAX(1,(T1+T)/2);
  bin = Threshold3(scn, T1, INT_MAX);
  ComputeSceneMBB(bin, Vm, VM);
  DestroyScene(&bin);

  //MSP must be in x/2:
  dx1 = Vm->x;
  dx2 = scn->xsize -1 - VM->x;
  if(dx1<dx2) VM->x = (scn->xsize -1 -dx1);
  else        Vm->x = dx2;

  //size must be odd numbers:
  zsize = VM->z - Vm->z + 1;
  ysize = VM->y - Vm->y + 1;
  if(zsize%2==0){
    if(Vm->z > 0) (Vm->z)--;
    else if(VM->z < scn->zsize-1) (VM->z)++;
  }
  if(ysize%2==0){
    if(Vm->y > 0) (Vm->y)--;
    else if(VM->y < scn->ysize-1) (VM->y)++;
  }
}


Scene *MRI_FillObjInterfaces(Scene *label,
			     Scene *scn,
			     float r,
			     MRI_Info info){
  AdjRel3 *A = Spheric(r);
  AdjVxl  *N;
  int fx,fy,fz,dp;
  int p,q,i,Lmax,l,lmax,ni;
  int *frequency;
  Scene *flabel=NULL;
  BMap *interface=NULL;
  //int Tgm_dark = ((info.Tcsf+info.Ecsf)+info.Tgm)/2;

  interface = BMapNew(scn->n);
  flabel = CopyScene(label);
  Lmax = MaximumValue3(label);
  frequency = (int *)calloc(Lmax+1,sizeof(int));

  N = AdjVoxels(label, A);
  FrameSizes3(A, &fx, &fy, &fz);
  dp  = fx*1;
  dp += fy*label->xsize;
  dp += fz*label->xsize*label->ysize;

  for(p=dp; p<label->n-dp; p++){
    if(scn->data[p]>=(info.Tcsf+info.Ecsf) &&
       label->data[p]==0){ //Brain tissue unlabeled.

      l = NIL;
      ni = 0;
      for(i=1; i<N->n; i++){
	q = p + N->dp[i];
	if(label->data[q]>0){
	  if(l==NIL){ 
	    ni = 1;
	    l = label->data[q];
	  }
	  else if(label->data[q]!=l){
	    ni = 2;
	    break;
	  }
	}
      }
      
      if(ni<2) continue;

      //interface between at least two objects.
      for(i=0; i<N->n; i++){
	q = p + N->dp[i];
	if(scn->data[q]>=(info.Tcsf+info.Ecsf) &&
	   label->data[q]==0){ //Brain tissue unlabeled.
	  _fast_BMapSet1(interface,q);
	}
      }
    }
  }

  for(p=dp; p<label->n-dp; p++){
    if(_fast_BMapGet(interface,p)>0){
      
      memset(frequency, 0, (Lmax+1)*sizeof(int));

      for(i=1; i<N->n; i++){
	q = p + N->dp[i];
	frequency[label->data[q]]++;
      }

      lmax = 1;
      for(l=1; l<=Lmax; l++){
	if(frequency[l]>frequency[lmax])
	  lmax = l;
      }
      if(frequency[lmax]>0)
	flabel->data[p] = lmax;
    }
  }
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&N);
  BMapDestroy(interface);
  free(frequency);

  return flabel;
}


