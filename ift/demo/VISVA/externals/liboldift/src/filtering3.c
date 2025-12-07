#include "filtering3.h"
#include "segmentation3.h"
#include "sort.h"

/**
 * Comparison function to be used in QuickSort
 */
static int comp(  void *a,   void *b) {
  return *(int*)a - *(int*)b;
}

Kernel3 *MakeKernel3(char *coefs)
{
  Kernel3 *K;
  AdjRel3 *A;
  int xsize,ysize,zsize,i;

  sscanf(coefs,"%d",&xsize);
  coefs=strchr(coefs,',')+1;
  sscanf(coefs,"%d",&ysize);
  coefs=strchr(coefs,',')+1;
  sscanf(coefs,"%d",&zsize);
  coefs=strchr(coefs,',')+1;

  A = Cube(xsize, ysize, zsize);
  K = CreateKernel3(A);
  for (i=0;i<A->n;i++) {
    sscanf(coefs,"%f",&K->val[i]);
    coefs=strchr(coefs,',')+1;
  }
  DestroyAdjRel3(&A);
  return(K);
}



Kernel3 *CreateKernel3(AdjRel3 *A)
{
  Kernel3 *K=NULL;
  int i;
  Voxel max, min;

  max.x = max.y = max.z = INT_MIN;
  min.x = min.y = min.z = INT_MAX;

  K = (Kernel3 *) calloc(1,sizeof(Kernel3));
  if (K == NULL){
    Error(MSG1,"CreateKernel3");
  }
  K->val = AllocFloatArray(A->n);
  K->adj = CreateAdjRel3(A->n);

  for (i=0;i<A->n;i++) {
    max.x = MAX(A->dx[i],max.x);
    max.y = MAX(A->dy[i],max.y);
    max.z = MAX(A->dz[i],max.z);
    min.x = MIN(A->dx[i],min.x);
    min.y = MIN(A->dy[i],min.y);
    min.z = MIN(A->dz[i],min.z);
    K->adj->dx[i] = A->dx[i];
    K->adj->dy[i] = A->dy[i];
    K->adj->dz[i] = A->dz[i];
  }
  
  K->xsize = max.x - min.x + 1;
  K->ysize = max.y - min.y + 1;
  K->zsize = max.z - min.z + 1;
  
  return(K);
}


Kernel3 *CloneKernel3(Kernel3 *K){
  Kernel3 *C;

  C = (Kernel3 *) calloc(1,sizeof(Kernel3));
  if(C == NULL)
    Error(MSG1,"CloneKernel3");

  C->val = AllocFloatArray(K->adj->n);
  memcpy(C->val, K->val,
	 sizeof(float)*K->adj->n);
  C->adj   = CloneAdjRel3(K->adj);
  C->xsize = K->xsize;
  C->ysize = K->ysize;
  C->zsize = K->zsize;

  return C;
}


Kernel3 *NormalizeKernel3(Kernel3 *K){
  Kernel3 *C = CloneKernel3(K);
  float wt=0.0;
  int i;
  
  for(i=0; i<K->adj->n; i++)
    wt += K->val[i];
  for(i=0; i<C->adj->n; i++)
    C->val[i] /= wt;

  return C;
}


void DestroyKernel3(Kernel3 **K)
{
  Kernel3 *aux;

  aux = *K;
  if(aux != NULL){
    if (aux->val != NULL)   free(aux->val); 
    DestroyAdjRel3(&(aux->adj));
    free(aux);    
    *K = NULL;
  }
}

/* Reflect kernel around x y, and z */

Kernel3 *FoldKernel3(Kernel3 *K)
{
  Kernel3 *Kf;
  int i,j;

  Kf = CreateKernel3(K->adj);
  for (i=0,j=Kf->adj->n-1; i < Kf->adj->n; i++,j--){
    Kf->val[i]     = K->val[j];
    Kf->adj->dx[i] = -K->adj->dx[j];
    Kf->adj->dy[i] = -K->adj->dy[j];
    Kf->adj->dz[i] = -K->adj->dz[j];
  }
  return(Kf);
}

Scene *LinearFilter3(Scene *scn, Kernel3 *K) /* generic kernel */ 
{
  Scene *cscn;
  Voxel u,v;
  AdjVxl *vxl;
  int p,q,i;
  float conv;
  
  cscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);

  vxl = AdjVoxels(scn,K->adj);

  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++){
	p = u.x + cscn->tby[u.y] + cscn->tbz[u.z];
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
	//cscn->data[p] = (int)(conv/K->adj->n);  // errado
	cscn->data[p] = (int)conv;  
      }

  DestroyAdjVxl(&vxl);
  return(cscn);
}

Scene *SobelFilter3(Scene *scn)
{
  Kernel3 *K;
  Scene *gx, *gy, *gz;
  int p,n=scn->xsize*scn->ysize*scn->zsize;  

  K = MakeKernel3("3,3,3,-1,0,1,-2,0,2,-1,0,1,-1,0,1,-2,0,2,-1,0,1,-1,0,1,-2,0,2,-1,0,1");
  gx = LinearFilter3(scn,K);
  DestroyKernel3(&K);
  K = MakeKernel3("3,3,3,-1,-2,-1,0,0,0,1,2,1,-1,-2,-1,0,0,0,1,2,1,-1,-2,-1,0,0,0,1,2,1");
  gy = LinearFilter3(scn,K);
  DestroyKernel3(&K);
  K = MakeKernel3("3,3,3,-1,-1,-1,-1,-2,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,1,1,1,1,2,1,1,1,1");
  gz = LinearFilter3(scn,K);
  DestroyKernel3(&K);
  for (p=0; p < n; p++) 
    gz->data[p] = (int)sqrt(gx->data[p]*gx->data[p]+gy->data[p]*gy->data[p]+gz->data[p]*gz->data[p]);
  
  DestroyScene(&gx);
  DestroyScene(&gy);
  return(gz);
}

Scene *MedianFilter3(Scene *scn, AdjRel3 *A)
{
  int *val,n,i,p,q;
  Voxel u,v;
  Scene *med;

  val = AllocIntArray(A->n);
  med = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  for (u.z=0; u.z < scn->zsize; u.z++)  
    for (u.y=0; u.y < scn->ysize; u.y++)  
      for (u.x=0; u.x < scn->xsize; u.x++) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	n = 0;
	for (i=0; i < A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(scn,v.x,v.y,v.z)){
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    val[n] = scn->data[q];
	    n++;
	  }
	}
	SelectionSort(val,n,INCREASING);
	med->data[p] = val[n/2];
      }
  free(val);
  return(med);
}

/**
 * Mode Filter3 - get the most frequent voxel value of an adjacency.
 * If all values have the same frequency, get the central voxel value.
 */
Scene *ModeFilter3(Scene *scn, AdjRel3 *A)
{
  int *data,n,i,p,q,*odata;
  int mode=0,fmode,m,fm,mid,fmid=0;
  Voxel u,v;
  Scene *modescn;

  data = AllocIntArray(A->n);
  modescn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (u.z=0; u.z < scn->zsize; ++u.z)  
    for (u.y=0; u.y < scn->ysize; ++u.y)  
      for (u.x=0; u.x < scn->xsize; ++u.x) {
	p = u.x + scn->tby[u.y] + scn->tbz[u.z];
	n = 0;
	for (i=0; i < A->n; ++i) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(scn,v.x,v.y,v.z)){
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    data[n] = scn->data[q];
	    n++;
	  }
	}
	/* QuickSort is faster than BucketSort for n < 130  */
	if (n < 130) {
	  qsort(data, n, sizeof(int), comp);
	  odata = data;
	} else {
	  odata = BucketSort(data,n,INCREASING);
	}
	mode  = mid  = scn->data[p];
	fmode = fmid = 1;
	m     = odata[0];
	fm    = 1;
	for (i=1; i<n; ++i) {
	  if (odata[i] != m) {
	    if (m == mid)
	      fmid = fm;
	    if (fm >= fmode) {
	      mode  = m;
	      fmode = fm;
	    }
	    m  = odata[i];
	    fm = 1;
	  } else {
	    ++fm;
	  }
	}
	if (m == mid)
	  fmid = fm;
	if (fm >= fmode) {
	  mode  = m;
	  fmode = fm;
	}
	if (n >= 130) free(odata);
	
	modescn->data[p] = fmid == fmode ? mid : mode;
      }
  free(data);
  return(modescn);
}

Kernel3 *LinearKernel3(AdjRel3 *A)
{
  float dmax=0.,*d;
  Kernel3 *K;
  int i;

  K = CreateKernel3(A);
  d = AllocFloatArray(A->n);
  for (i=0;i<A->n;i++) {
    d[i] = sqrt(A->dx[i] * A->dx[i] + A->dy[i] * A->dy[i] + A->dz[i] * A->dz[i]);
    dmax = MAX(d[i],dmax);
  }
  dmax += 1.0; // avoid zero values inside kernel
  for (i=0;i<A->n;i++) {   
    K->val[i] = 1.0 - (d[i] / (dmax)); 
  }
  free(d);
  return(K);
}

/* stddev = 1.0 -> erf = 68.2% */
/* stddev = 2.0 -> erf = 95.4% */
/* stddev = 3.0 -> erf = 99.7% */

Kernel3 *GaussianKernel3(AdjRel3 *A, float stddev)
{
  double k,k1,d2,sigma,sigma2;
  Kernel3 *K;
  int i;

  d2 = 0.0;
  for (i=0;i<A->n;i++) {
    d2=MAX((double)(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]+A->dz[i]*A->dz[i]),d2);
  }
  sigma2 = d2 / (stddev*stddev);
  sigma = sqrt(sigma2);
  k = 2.0*sigma2;
  k1 = 1.0/(sqrt(2*PI)*sigma);
  K = CreateKernel3(A);
  for (i=0;i<A->n;i++) {
    d2 = (double)(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]+A->dz[i]*A->dz[i]);
    K->val[i] = (float)(k1*exp(-d2/k)); // gaussian
  }
  return(K);
}

Kernel3 *LaplacianKernel3(AdjRel3 *A, float stddev)
{

  double k,k1,d2,sigma2;
  Kernel3 *K;
  int i;

  d2 = 0.0;
  for (i=0;i<A->n;i++) {
    d2=MAX((double)(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]+A->dz[i]*A->dz[i]),d2);
  }
  sigma2 = d2 / (stddev*stddev);
  k = 2.0*sigma2;
  k1 = 1.0/sqrt(2*PI);
  K = CreateKernel3(A);
  for (i=0;i<A->n;i++) {
    d2 = (double)(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]+A->dz[i]*A->dz[i]);
    K->val[i] = (float)((d2/(2*sigma2)+1.0)* k1 * exp(-d2/k)); // laplacian
  }
  return(K);

}

real *RealConvolution3D(real *value, Kernel3 *K,int xsize, int ysize,int zsize)
{
  real *cscn, conv;
  Voxel u,v;
  int p,i,q, xysize;
  Kernel3 *Kf;
  Scene scn;
  scn.xsize = xsize;
  scn.ysize = ysize;
  scn.zsize = zsize;
  xysize = xsize * ysize;

  cscn = AllocRealArray(xysize * zsize);
  Kf   = FoldKernel3(K);

  for (u.z=0; u.z < zsize; u.z++)
    for (u.y=0; u.y < ysize; u.y++)
      for (u.x=0; u.x < xsize; u.x++) {
	conv = 0.0;
	p = u.x + u.y * xsize + u.z * xysize;
	for (i=0; i < Kf->adj->n; i++){
	  v.x = u.x + Kf->adj->dx[i]; 
	  v.y = u.y + Kf->adj->dy[i]; 
	  v.z = u.z + Kf->adj->dz[i]; 
	  if (ValidVoxel(&scn,v.x,v.y,v.z)){
	    q = v.x + v.y * xsize + v.z * xysize;
	    conv += value[q] * Kf->val[i];
	  }	
	}
	cscn[p] = conv;
      }

  DestroyKernel3(&Kf);
  return(cscn);

}
