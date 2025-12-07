#include "preproc.h"
#include <pthread.h>


void SuppressHighIntensities(Scene *scn){
  Curve *hist;
  int T,p,n;

  hist = Histogram3(scn);
  T = AreaPercentageHigherThreshold(hist, 99.95);
  n = scn->xsize*scn->ysize*scn->zsize;
  for(p=0; p<n; p++)
    if(scn->data[p]>T)
      scn->data[p] = T;
  DestroyCurve(&hist);
}



int MeanAboveThreshold(Scene *scn, int T){
  float sum=0.0;
  int nsum=0;
  int p,val,T2;

  for(p=0; p<scn->n; p++){
    val = scn->data[p];
    if(val>T){
      sum += (float)val;
      nsum++;
    }
  }
  T2 = ROUND(sum/((float)nsum));
  return T2;
}


int SceneMeanInMask(Scene *scn, Scene *mask){
  float sum=0.0;
  int nsum=0;
  int p,T=0;

  for(p=0; p<scn->n; p++){
    if(mask->data[p]>0){
      sum += (float)scn->data[p];
      nsum++;
    }
  }
  if(nsum>0) T = ROUND(sum/((float)nsum));
  return T;
}


int SceneRangeMeanValue(Scene *scn, int lower, int higher){
  float sum=0.0;
  int nsum=0,p,val;
  for(p=0; p<scn->n; p++){
    val = scn->data[p];
    if(val>=lower && val<=higher){
      sum += (float)val;
      nsum++;
    }
  }
  return ROUND(sum/((float)nsum));
}


void MeansAboveBelowT(Scene *scn, int T, int *T1, int *T2){
  long long mean1=0,mean2=0,nv1=0,nv2=0;
  int p,n,delta;

  n  = scn->xsize*scn->ysize*scn->zsize;
  for(p=0; p < n; p++) {
    if (scn->data[p]<T){
      mean1+= scn->data[p];
      nv1++;
    }
    if (scn->data[p]>T){
      mean2+= scn->data[p];
      nv2++;
    }
  }

  delta = (int)(mean2/nv2 - mean1/nv1)/2;
  *T1   = T+delta;
  *T2   = T-delta;

  //printf("T %d T1 %d T2 %d delta %d\n",T,*T1,*T2,delta);
}


Scene *ApplySShape(Scene *scn, int a, int b, int c){
  Scene *enha=NULL;
  int p,n;
  float weight;

  enha = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n    = scn->xsize*scn->ysize*scn->zsize;

  for (p=0; p < n; p++) {
    if (scn->data[p]<=a){
      weight = 0.0;
    }else{
      if ((scn->data[p]>a)&&(scn->data[p]<=b)){
	weight = (2.0*((float)scn->data[p]-a)*((float)scn->data[p]-a)/(((float)c-a)*((float)c-a)));
      }else{
	if ((scn->data[p]>b)&&(scn->data[p]<=c)){
	  weight = (1.0 - 2.0*((float)scn->data[p]-c)*((float)scn->data[p]-c)/(((float)c-a)*((float)c-a)));
	}else{
	  weight = 1.0; 
	}
      }
      
    }
    enha->data[p]=(int)(scn->data[p]*weight);
  }
  return(enha);
}


int ComputeOtsu3(Scene *scn){
  Curve *hist=NormHistogram3(scn);
  double p1,p2,m1,m2,s1,s2,J,Jmax=-1.0;
  int i,T,Topt=0,Imax=MaximumValue3(scn);

  for (T=1; T < Imax; T++){
    p1 = 0.0;
    for (i=0; i <= T; i++) 
      p1 += hist->Y[i];
    p2 = 1.0 - p1;
    if ((p1 > 0.0)&&(p2 > 0.0)){
      m1 = 0.0;
      for (i=0; i <= T; i++) 
	m1 += hist->Y[i]*i;
      m1 /= p1;
      m2 = 0.0;
      for (i=T+1; i <= Imax; i++) 
	m2 += hist->Y[i]*i;
      m2 /= p2;
      s1 = 0.0;
      for (i=0; i <= T; i++) 
	s1 += hist->Y[i]*(i-m1)*(i-m1);
      s1 /= p1;
      s2 = 0.0;
      for (i=T+1; i <= Imax; i++) 
	s2 += hist->Y[i]*(i-m2)*(i-m2);
      s2 /= p2;
      J = (p1*p2*(m1-m2)*(m1-m2))/(p1*s1+p2*s2);
    }else{
      J = 0.0;      
    }
    if (J > Jmax){
      Jmax = J;
      Topt = T;
    }
  }
  //printf("Otsu: %d\n",Topt);

  DestroyCurve(&hist);
  return(Topt);
}


int ComputeOtsuMask3(Scene *scn, Scene *mask){
  Curve *hist=NormHistogramMask3(scn, mask);
  double p1,p2,m1,m2,s1,s2,J,Jmax=-1.0;
  int i,T,Topt=0,Imax=MaximumValueMask3(scn, mask);

  for (T=1; T < Imax; T++){
    p1 = 0.0;
    for (i=0; i <= T; i++) 
      p1 += hist->Y[i];
    p2 = 1.0 - p1;
    if ((p1 > 0.0)&&(p2 > 0.0)){
      m1 = 0.0;
      for (i=0; i <= T; i++) 
	m1 += hist->Y[i]*i;
      m1 /= p1;
      m2 = 0.0;
      for (i=T+1; i <= Imax; i++) 
	m2 += hist->Y[i]*i;
      m2 /= p2;
      s1 = 0.0;
      for (i=0; i <= T; i++) 
	s1 += hist->Y[i]*(i-m1)*(i-m1);
      s1 /= p1;
      s2 = 0.0;
      for (i=T+1; i <= Imax; i++) 
	s2 += hist->Y[i]*(i-m2)*(i-m2);
      s2 /= p2;
      J = (p1*p2*(m1-m2)*(m1-m2))/(p1*s1+p2*s2);
    }else{
      J = 0.0;      
    }
    if (J > Jmax){
      Jmax = J;
      Topt = T;
    }
  }
  //printf("Otsu: %d\n",Topt);

  DestroyCurve(&hist);
  return(Topt);
}


/* Compute mean and stdev of the input scene */
void   DescriptiveStatisticsMask3(Scene *scn,
				  Scene *mask,
				  real *mean,
				  real *stdev){
  double sum,sum2,num,val;
  int n, p;

  n = scn->xsize*scn->ysize*scn->zsize;
  num = sum = sum2 = 0.0;
  for(p=0; p<n; p++){
    if(mask->data[p]>0){
      val = (double)scn->data[p];
      sum  += val;
      sum2 += val*val;
      num++;
    }
  }

  *mean = sum/num;
  *stdev = sum2-(sum*sum)/num;
  *stdev = sqrt(*stdev/num);
}


void   OrderStatisticMask3(Scene *scn,
			   Scene *mask,
			   real *median){
  Curve *hist = NULL;
  int count,n,i,j;

  hist = HistogramMask3(scn, mask);
  n = 0;
  for(i=0; i<hist->n; i++)
    n += ROUND(hist->Y[i]);

  i = -1;
  count = 0;
  while(count<n/2){
    i++;
    count += ROUND(hist->Y[i]);
  }
  if(n%2>0){
    if(count==n/2){
      j = i+1;
      while(ROUND(hist->Y[j])==0)
	j++;
      *median = (real)j;
    }
    else
      *median = (real)i;
  }
  else{
    if(count==n/2){
      j = i+1;
      while(ROUND(hist->Y[j])==0)
	j++;
      *median = (real)(i+j)/2.0; 
    }
    else
      *median = (real)i;
  }
  
  DestroyCurve(&hist);
}


Scene *Convolution3(Scene *scn, Kernel3 *K){
  Scene *cscn;
  Voxel u,v;
  AdjVxl *vxl;
  int p,q,i;
  float conv;
  
  cscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  cscn->dx = scn->dx;
  cscn->dy = scn->dy;
  cscn->dz = scn->dz;
  vxl = AdjVoxels(scn,K->adj);

  for(u.z=0; u.z<scn->zsize; u.z++)
    for(u.y=0; u.y<scn->ysize; u.y++)
      for(u.x=0; u.x<scn->xsize; u.x++){
	p = VoxelAddress(cscn,u.x,u.y,u.z);
	conv=0.0;
	for (i=0;i<K->adj->n;i++) {
	  v.x = u.x + K->adj->dx[i]; 
	  v.y = u.y + K->adj->dy[i];
          v.z = u.z + K->adj->dz[i];
	  if (ValidVoxel(scn,v.x,v.y,v.z)){
	    q = p + vxl->dp[i];
	    conv += ((float)scn->data[q])*K->val[i];	   
	  }      
	}
	cscn->data[p] = ROUND(conv);
      }

  DestroyAdjVxl(&vxl);
  return(cscn);
}


Scene *OptConvolution3(Scene *scn, Kernel3 *K){
  Scene *cscn;
  AdjVxl *vxl;
  int p,q,i,N,dx,dy,dz,dn;
  float conv;
  
  N = scn->xsize*scn->ysize*scn->zsize;
  cscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  cscn->dx = scn->dx;
  cscn->dy = scn->dy;
  cscn->dz = scn->dz;
  vxl = AdjVoxels(scn,K->adj);

  dx = dy = dz = 0;
  for(i=0;i<K->adj->n;i++){
    dx = MAX(dx, abs(K->adj->dx[i]));
    dy = MAX(dy, abs(K->adj->dy[i]));
    dz = MAX(dz, abs(K->adj->dz[i]));
  }

  dx *= 1;
  dy *= scn->xsize;
  dz *= scn->xsize * scn->ysize;
  dn = dx+dy+dz;

  for(p=dn; p<N-dn; p++){
    conv=0.0;
    for(i=0; i<K->adj->n; i++){
      q = p + vxl->dp[i];
      conv += ((float)scn->data[q])*K->val[i];	   
    }
    cscn->data[p] = ROUND(conv);
  }

  DestroyAdjVxl(&vxl);
  return(cscn);
}



Kernel3 *SphericalGaussianKernel3(float R, float s, float f)
{
  float R2,r2;
  Kernel3 *K;
  AdjRel3 *A;
  int i;

  R2 = R*R;
  A = Spheric(R);
  K = CreateKernel3(A);
  for (i=0;i<A->n;i++) {
    r2 = A->dx[i] * A->dx[i] + A->dy[i] * A->dy[i] + A->dz[i] * A->dz[i];
    K->val[i] = s*exp (-f*(r2/R2));
  }
  DestroyAdjRel3(&A);
  return(K);

}


Scene   *GaussianBlur3(Scene *scn){
  Scene *blur;
  Kernel3 *K,*NK;

  K  = SphericalGaussianKernel3(1.0, 10.0, 1.0); //2.0, 10.0, 1.0
  NK = NormalizeKernel3(K);
  blur = Convolution3(scn, NK);
  SetVoxelSize(blur, scn->dx, scn->dy, scn->dz);
  DestroyKernel3(&NK);
  DestroyKernel3(&K);
  return blur;
}

Scene   *FastGaussianBlur3(Scene *scn){
  Scene *blur;
  Kernel3 *K,*NK;

  K  = SphericalGaussianKernel3(1.0, 10.0, 1.0); //2.0, 10.0, 1.0
  NK = NormalizeKernel3(K);
  blur = FastConvolution3(scn, NK);
  SetVoxelSize(blur, scn->dx, scn->dy, scn->dz);
  DestroyKernel3(&NK);
  DestroyKernel3(&K);
  return blur;
}

Scene   *OptGaussianBlur3(Scene *scn){
  Scene *blur;
  Kernel3 *K,*NK;

  K  = SphericalGaussianKernel3(1.0, 10.0, 1.0); //2.0, 10.0, 1.0
  NK = NormalizeKernel3(K);
  blur = OptConvolution3(scn, NK);
  SetVoxelSize(blur, scn->dx, scn->dy, scn->dz);
  DestroyKernel3(&NK);
  DestroyKernel3(&K);
  return blur;
}


Scene   *FastOptGaussianBlur3(Scene *scn){
  Scene *blur;
  Kernel3 *K,*NK;
  
  K  = SphericalGaussianKernel3(1.0, 10.0, 1.0); //2.0, 10.0, 1.0
  NK = NormalizeKernel3(K);
  blur = FastOptConvolution3(scn, NK);
  SetVoxelSize(blur, scn->dx, scn->dy, scn->dz);
  DestroyKernel3(&NK);
  DestroyKernel3(&K);
  return blur;
}


Scene   *Subsampling3(Scene *scn){
  Scene *scl;
  Voxel u,v;
  int p,q;
  int xsize,ysize,zsize;

  xsize = (int)(scn->xsize/2) + (scn->xsize%2);
  ysize = (int)(scn->ysize/2) + (scn->ysize%2);
  zsize = (int)(scn->zsize/2) + (scn->zsize%2);
  if(xsize%2==0) xsize++;
  if(ysize%2==0) ysize++;
  if(zsize%2==0) zsize++;
  scl = CreateScene(xsize, ysize, zsize);
  scl->dx = 2.0*scn->dx;
  scl->dy = 2.0*scn->dy;
  scl->dz = 2.0*scn->dz;

  for(v.z=0; v.z<scl->zsize; v.z++)
    for(v.y=0; v.y<scl->ysize; v.y++)
      for(v.x=0; v.x<scl->xsize; v.x++){
	p = VoxelAddress(scl,v.x,v.y,v.z);
	u.x = (v.x-scl->xsize/2)*2 + scn->xsize/2;
	u.y = (v.y-scl->ysize/2)*2 + scn->ysize/2;
	u.z = (v.z-scl->zsize/2)*2 + scn->zsize/2;

	if(ValidVoxel(scn,u.x,u.y,u.z)){
	  q = VoxelAddress(scn,u.x,u.y,u.z);
	  scl->data[p] = scn->data[q];
	}
      }
  
  return(scl);
}


//-----------------------------------
//-----------------------------------
//-----------------------------------

Scene *MeanLocalIFT3D(Scene *scn, Scene *mask, int rsize){
  Scene *fscn=CopyScene(scn);
  Scene *cost;
  AdjRel3 *A=Spheric(1.0);
  GQueue *Q=NULL;
  Voxel u,v;
  int Imax,edge,cst;
  int n,p,s,t,i,ki,val_p;
  int *px = (int *)calloc(rsize,sizeof(int));
  int mean;

  n    = scn->xsize*scn->ysize*scn->zsize;
  cost = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  SetScene(cost, INT_MAX);
  Imax = MaximumValue3(scn);
  Q    = CreateGQueue(Imax+1, n, cost->data);

  for(p=0; p<n; p++){
    if(mask->data[p]==0) continue;
    val_p = scn->data[p];
    cost->data[p] = 0;
    InsertGQueue(&Q,p);

    mean = 0;
    for(ki=0; ki<rsize; ki++){
      s = RemoveGQueue(Q);
      mean += scn->data[s];
      px[ki] = s;
      u.x = VoxelX(scn, s);
      u.y = VoxelY(scn, s);
      u.z = VoxelZ(scn, s);
      for(i=1; i<A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	v.z = u.z + A->dz[i];
	if(ValidVoxel(scn, v.x, v.y, v.z)){
	  t = VoxelAddress(scn, v.x, v.y, v.z);
	  if(Q->L.elem[t].color != BLACK){	  
	    edge = abs(scn->data[t] - val_p);
	    cst  = cost->data[s]+edge; //MAX(cost->data[s],edge);
	    if(cst < cost->data[t]){
	      if(Q->L.elem[t].color == GRAY)
		RemoveGQueueElem(Q,t); // color of t = BLACK
	      cost->data[t] = cst;
	      InsertGQueue(&Q,t);
	    }
	  }
	}
      }
    }
    fscn->data[p] = mean/rsize;

    /* Reset queue manually */
    for(ki=0; ki<rsize; ki++){
      s = px[ki];
      cost->data[s] = INT_MAX;
      Q->L.elem[s].color = WHITE;
    }
    while (!EmptyGQueue(Q)){
      s = RemoveGQueue(Q);
      cost->data[s] = INT_MAX;
      Q->L.elem[s].color = WHITE;
    }
    Q->C.minvalue = INT_MAX;
    Q->C.maxvalue = INT_MIN;
  }

  DestroyScene(&cost);
  DestroyAdjRel3(&A);
  DestroyGQueue(&Q);
  free(px);

  return fscn;
}


Scene *MedianLocalIFT(Scene *scn, Scene *mask, int rsize){
  int     Imax,i,z,p,q,n=scn->xsize*scn->ysize;
  Pixel   u,v;
  AdjRel *A4=Circular(1.0);
  GQueue *Q=NULL;
  int     cst,weight,nv,r;
  Image  *cost=CreateImage(scn->xsize,scn->ysize);
  Set    *pts=NULL;
  Scene  *fscn=CopyScene(scn);
  int    *V=AllocIntArray(rsize);

  Imax = MaximumValue3(scn);
  Q    = CreateGQueue(Imax+1,n,cost->val);

  for (p=0; p < n; p++){ 
    cost->val[p]=INT_MAX;
  } 

  /* Compute mean filter by local IFTs in a slice-by-slice fashion */

  for (z=0; z < scn->zsize; z++){

    //fprintf(stdout,"Processing slice %d out of %d\n",z+1,scn->zsize);

    for (r=0; r < n; r++){

      if (mask->data[r+scn->tbz[z]]>0) { /* avoid computation on background */

	nv=0;
	cost->val[r]=0;
	InsertGQueue(&Q,r);
	InsertSet(&pts,r);

	while(!EmptyGQueue(Q)) {
	  p=RemoveGQueue(Q);
	  
	  if (nv < rsize){
	    V[nv]=scn->data[p+scn->tbz[z]];
	    i = nv;
	    while((i>0)&&(V[i]<V[i-1])){
	      cst = V[i-1];
	      V[i-1]=V[i];
	      V[i]=cst;
	      i--;
	    }
	  }else{
	    fscn->data[r+fscn->tbz[z]] = V[rsize/2];
	    break;
	  }
	  nv++;

	  u.x = p%scn->xsize;
	  u.y = p/scn->xsize;

	  for (i=1; i < A4->n; i++){
	    v.x = u.x + A4->dx[i];
	    v.y = u.y + A4->dy[i];
	    if (ValidPixel(cost,v.x,v.y)){
	      q = v.x + cost->tbrow[v.y];
	      if (Q->L.elem[q].color != BLACK){
		weight = fabs(scn->data[q+scn->tbz[z]] - scn->data[r+scn->tbz[z]]);
		cst = (cost->val[p]+weight);
		if(cst < cost->val[q]){
		  if(Q->L.elem[q].color == GRAY)
		    RemoveGQueueElem(Q,q);	    
		  cost->val[q]  = cst;
		  InsertGQueue(&Q,q);
		  InsertSet(&pts,q);
		}
	      }
	    }
	  }
	}

	Q->C.minvalue = INT_MAX;
	Q->C.maxvalue = INT_MIN;
	for (i=0; i < Q->C.nbuckets+1; i++)
	  Q->C.first[i]=Q->C.last[i]=NIL;
	while(pts != NULL){
	  p=RemoveSet(&pts);
	  Q->L.elem[p].next  =  Q->L.elem[p].prev = NIL;
	  Q->L.elem[p].color = WHITE;
	  cost->val[p]=INT_MAX;
	}
      }
    }
  }

  DestroyGQueue(&Q);
  free(V);
  DestroyImage(&cost);
  DestroyAdjRel(&A4);
  
  return(fscn);
}


Scene *MeanLocalIFT(Scene *scn, Scene *mask, int rsize){
  int     Imax,i,z,p,q,n=scn->xsize*scn->ysize;
  Pixel   u,v;
  AdjRel *A4=Circular(1.0);
  GQueue *Q=NULL;
  int     cst,weight,nv,r;
  Image  *cost=CreateImage(scn->xsize,scn->ysize);
  Set    *pts=NULL;
  Scene  *fscn=CopyScene(scn);

  Imax = MaximumValue3(scn);
  Q    = CreateGQueue(Imax+1,n,cost->val);

  for (p=0; p < n; p++){ 
    cost->val[p]=INT_MAX;
  } 

  /* Compute mean filter by local IFTs in a slice-by-slice fashion */

  for (z=0; z < scn->zsize; z++){

    //fprintf(stdout,"Processing slice %d out of %d\n",z+1,scn->zsize);

    for (r=0; r < n; r++){

      if (mask->data[r+scn->tbz[z]]>0) { /* avoid computation on background */
	fscn->data[r+scn->tbz[z]] = 0;
	nv=0;
	cost->val[r]=0;
	InsertGQueue(&Q,r);
	InsertSet(&pts,r);

	while(!EmptyGQueue(Q)) {
	  p=RemoveGQueue(Q);
	  
	  if (nv < rsize){
	   fscn->data[r+fscn->tbz[z]] += scn->data[p+scn->tbz[z]];	
	  }else{
	    fscn->data[r+fscn->tbz[z]] /= rsize;
	    break;
	  }
	  nv++;

	  u.x = p%scn->xsize;
	  u.y = p/scn->xsize;

	  for (i=1; i < A4->n; i++){
	    v.x = u.x + A4->dx[i];
	    v.y = u.y + A4->dy[i];
	    if (ValidPixel(cost,v.x,v.y)){
	      q = v.x + cost->tbrow[v.y];
	      if (Q->L.elem[q].color != BLACK){
		weight = fabs(scn->data[q+scn->tbz[z]] - scn->data[r+scn->tbz[z]]);
		cst = (cost->val[p]+weight);
		if(cst < cost->val[q]){
		  if(Q->L.elem[q].color == GRAY)
		    RemoveGQueueElem(Q,q);	    
		  cost->val[q]  = cst;
		  InsertGQueue(&Q,q);
		  InsertSet(&pts,q);
		}
	      }
	    }
	  }
	}

	Q->C.minvalue = INT_MAX;
	Q->C.maxvalue = INT_MIN;
	for (i=0; i < Q->C.nbuckets+1; i++)
	  Q->C.first[i]=Q->C.last[i]=NIL;
	while(pts != NULL){
	  p=RemoveSet(&pts);
	  Q->L.elem[p].next  =  Q->L.elem[p].prev = NIL;
	  Q->L.elem[p].color = WHITE;
	  cost->val[p]=INT_MAX;
	}
      }
    }
  }

  DestroyGQueue(&Q);
  DestroyImage(&cost);
  DestroyAdjRel(&A4);
  
  return(fscn);
}


Scene *WeightMeanLocalIFT(Scene *scn, Scene *mask, int rsize){
  int     Imax,i,z,p,q,n=scn->xsize*scn->ysize;
  Pixel   u,v;
  AdjRel *A4=Circular(1.0);
  GQueue *Q=NULL;
  int     cst,weight,nv,r;
  Image  *cost=CreateImage(scn->xsize,scn->ysize);
  Set    *pts=NULL;
  Scene  *fscn=CopyScene(scn);
  float   mean,w,sw,fval,x,K;

  Imax = MaximumValue3(scn);
  K = 2.0*(0.04*Imax*Imax);
  Q    = CreateGQueue(Imax+1,n,cost->val);

  for (p=0; p < n; p++){ 
    cost->val[p]=INT_MAX;
  } 

  /* Compute mean filter by local IFTs in a slice-by-slice fashion */

  for (z=0; z < scn->zsize; z++){

    //fprintf(stdout,"Processing slice %d out of %d\n",z+1,scn->zsize);

    for (r=0; r < n; r++){

      if (mask->data[r+scn->tbz[z]]>0) { /* avoid computation on background */

	nv=0;
	cost->val[r]=0;
	InsertGQueue(&Q,r);
	InsertSet(&pts,r);
	x=sw=fval=0;

	while(!EmptyGQueue(Q)) {
	  p=RemoveGQueue(Q);
	  
	  if (nv < rsize){
	    x  += scn->data[p+scn->tbz[z]];
	    mean = x/(nv+1);
	    w = exp(-(scn->data[p+scn->tbz[z]]-mean)*(scn->data[p+scn->tbz[z]]-mean)/K);
	    sw += w;
	    fval += (w*scn->data[p+scn->tbz[z]]);	
	  }else{
	    fval /= sw;
	    fscn->data[r+fscn->tbz[z]] = (int)fval;
	    break;
	  }
	  nv++;

	  u.x = p%scn->xsize;
	  u.y = p/scn->xsize;

	  for (i=1; i < A4->n; i++){
	    v.x = u.x + A4->dx[i];
	    v.y = u.y + A4->dy[i];
	    if (ValidPixel(cost,v.x,v.y)){
	      q = v.x + cost->tbrow[v.y];
	      if (Q->L.elem[q].color != BLACK){
		weight = fabs(scn->data[q+scn->tbz[z]] - scn->data[r+scn->tbz[z]]);
		cst = (cost->val[p]+weight);
		if(cst < cost->val[q]){
		  if(Q->L.elem[q].color == GRAY)
		    RemoveGQueueElem(Q,q);	    
		  cost->val[q]  = cst;
		  InsertGQueue(&Q,q);
		  InsertSet(&pts,q);
		}
	      }
	    }
	  }
	}

	Q->C.minvalue = INT_MAX;
	Q->C.maxvalue = INT_MIN;
	for (i=0; i < Q->C.nbuckets+1; i++)
	  Q->C.first[i]=Q->C.last[i]=NIL;
	while(pts != NULL){
	  p=RemoveSet(&pts);
	  Q->L.elem[p].next  =  Q->L.elem[p].prev = NIL;
	  Q->L.elem[p].color = WHITE;
	  cost->val[p]=INT_MAX;
	}
      }
    }
  }

  DestroyGQueue(&Q);
  DestroyImage(&cost);
  DestroyAdjRel(&A4);
  
  return(fscn);
}



//==========================================



Scene *InfRecMarker(Scene *iscn, Scene *marker){
  Voxel u,v;
  int i,s,p,q,n,tmp,Imax;
  Scene *cost;
  GQueue *Q;
  AdjRel3 *A=Spheric(1.8);

  Imax = MaximumValue3(iscn);
  cost = CreateScene(iscn->xsize,iscn->ysize,iscn->zsize);
  s    = iscn->xsize*iscn->ysize;
  n    = s*iscn->zsize;
  Q    = CreateGQueue(Imax+1,n,cost->data);

  for (p=0; p < n; p++) {
    cost->data[p]=INT_MAX;
    if (marker->data[p]!=0){
      cost->data[p]=Imax-iscn->data[p];
      InsertGQueue(&Q,p);
    }      
  }
  
  while (!EmptyGQueue(Q)){
    p   = RemoveGQueue(Q);
    u.z =  p / s;
    i   = (p % s);
    u.y = i / iscn->xsize;
    u.x = i % iscn->xsize;
    for (i = 1; i < A->n; i++) {
      v.z = u.z + A->dz[i];
      v.y = u.y + A->dy[i];
      v.x = u.x + A->dx[i];
      if (ValidVoxel(iscn, v.x, v.y, v.z)) {
	q = v.x + iscn->tby[v.y] + iscn->tbz[v.z];
	if (cost->data[q] > cost->data[p]){
	  tmp = MAX(cost->data[p],Imax-iscn->data[q]);
	  if (tmp < cost->data[q]){
	    if (cost->data[q]==INT_MAX){
	      cost->data[q] = tmp;
	      InsertGQueue(&Q,q);
	    }else
	      UpdateGQueue(&Q,q,tmp);
	  }
	}
      }
    }
  }

  DestroyGQueue(&Q);
  DestroyAdjRel3(&A);

  return(cost);  
}


//-----------------------------------------

Scene *FastConvolution3(Scene *scn, Kernel3 *K){
  ArgConvolution3 arg;
  ArgConvolution3 args[8];
  int i,nprocs;
  int first,last,nelems,de;
  pthread_t thread_id[8];
  int iret[8];

  nprocs = GetNumberOfProcessors();
  //printf("nprocs: %d\n",nprocs);

  if(nprocs<=1) return Convolution3(scn, K);
  if(nprocs>=8) nprocs = 8;

  arg.scn  = scn;
  arg.cscn = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  (arg.cscn)->dx = scn->dx;
  (arg.cscn)->dy = scn->dy;
  (arg.cscn)->dz = scn->dz;
  arg.vxl  = AdjVoxels(scn,K->adj);
  arg.K    = K;

  first  = 0;
  last   = scn->n-1;
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
			     ThreadConvolution3,
			     (void*)&args[i]);
    arg = args[i];
  }
  
  //Wait till threads are complete before main continues.
  for(i=0; i<nprocs; i++){
    pthread_join(thread_id[i], NULL);
  }

  DestroyAdjVxl(&(arg.vxl));
  return arg.cscn;
}



void  *ThreadConvolution3(void *arg){
  ArgConvolution3 *p_arg;
  Scene *cscn,*scn;
  Voxel u,v;
  AdjVxl *vxl;
  Kernel3 *K;
  int p,q,i;
  float conv;

  p_arg = (ArgConvolution3 *)arg;
  scn  = p_arg->scn;
  cscn = p_arg->cscn;
  vxl  = p_arg->vxl;
  K    = p_arg->K;

  for(p=p_arg->i; p<=p_arg->j; p++){
    u.x = VoxelX(cscn,p);
    u.y = VoxelY(cscn,p);
    u.z = VoxelZ(cscn,p);
    conv=0.0;
    for(i=0;i<K->adj->n;i++) {
      v.x = u.x + K->adj->dx[i]; 
      v.y = u.y + K->adj->dy[i];
      v.z = u.z + K->adj->dz[i];
      if(ValidVoxel(scn,v.x,v.y,v.z)){
	q = p + vxl->dp[i];
	conv += ((float)scn->data[q])*K->val[i];	   
      }      
    }
    cscn->data[p] = ROUND(conv);
  }
  return NULL;
}


Scene *FastOptConvolution3(Scene *scn, Kernel3 *K){
  ArgConvolution3 arg;
  ArgConvolution3 args[8];
  int i,nprocs;
  int first,last,nelems,de;
  pthread_t thread_id[8];
  int iret[8];
  int dx,dy,dz,dn;

  nprocs = GetNumberOfProcessors();
  //printf("nprocs: %d\n",nprocs);

  if(nprocs<=1) return OptConvolution3(scn, K);
  if(nprocs>=8) nprocs = 8;

  arg.scn  = scn;
  arg.cscn = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  (arg.cscn)->dx = scn->dx;
  (arg.cscn)->dy = scn->dy;
  (arg.cscn)->dz = scn->dz;
  arg.vxl  = AdjVoxels(scn,K->adj);
  arg.K    = K;

  dx = dy = dz = 0;
  for(i=0;i<K->adj->n;i++){
    dx = MAX(dx, abs(K->adj->dx[i]));
    dy = MAX(dy, abs(K->adj->dy[i]));
    dz = MAX(dz, abs(K->adj->dz[i]));
  }
  dx *= 1;
  dy *= scn->xsize;
  dz *= scn->xsize * scn->ysize;
  dn = dx+dy+dz;

  first  = dn;
  last   = scn->n-dn-1;
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
			     ThreadOptConvolution3,
			     (void*)&args[i]);
    arg = args[i];
  }
  
  //Wait till threads are complete before main continues.
  for(i=0; i<nprocs; i++){
    pthread_join(thread_id[i], NULL);
  }

  DestroyAdjVxl(&(arg.vxl));
  return arg.cscn;
}


void  *ThreadOptConvolution3(void *arg){
  ArgConvolution3 *p_arg;
  Scene *cscn,*scn;
  AdjVxl *vxl;
  Kernel3 *K;
  int p,q,i;
  float conv;

  p_arg = (ArgConvolution3 *)arg;
  scn  = p_arg->scn;
  cscn = p_arg->cscn;
  vxl  = p_arg->vxl;
  K    = p_arg->K;

  for(p=p_arg->i; p<=p_arg->j; p++){
    conv=0.0;
    for(i=0; i<K->adj->n; i++){
      q = p + vxl->dp[i];
      conv += ((float)scn->data[q])*K->val[i];	   
    }
    cscn->data[p] = ROUND(conv);
  }
  return NULL;
}


