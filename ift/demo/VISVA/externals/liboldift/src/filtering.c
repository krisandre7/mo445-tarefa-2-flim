#include "filtering.h"
#include "radiometric.h"
#include "sort.h"
#include "common.h"
#include "segmentation.h"
#include "analysis.h"

/**
 * Comparison function to be used in QuickSort
 */
static int comp(  void *a,   void *b) {
  return *(int*)a - *(int*)b;
}


Kernel *MakeKernel(char *coefs)
{
  Kernel *K;
  AdjRel *A;
  int xsize,ysize,i;
  float val;

  sscanf(coefs,"%d",&xsize);
  coefs=strchr(coefs,',')+1;
  sscanf(coefs,"%d",&ysize);
  coefs=strchr(coefs,',')+1;

  A = Box(xsize, ysize);
  K = CreateKernel(A);
  for (i=0;i<A->n;i++) {
    sscanf(coefs,"%f",&K->val[i]);
    coefs=strchr(coefs,',')+1;
  }

  /* Put the middle value (corresponding to the origin) at the first
     place to match the vector of coefficients with the vector of
     displacements (adjacency relation) */

  for (i=A->n/2; i > 0; i--){
    val = K->val[i];
    K->val[i]=K->val[i-1];
    K->val[i-1]=val;
  }

  DestroyAdjRel(&A);
  return(K);
}

Kernel *CreateKernel(AdjRel *A)
{
  Kernel *K=NULL;
  int i;
  Pixel max, min;

  max.x = max.y = INT_MIN;
  min.x = min.y = INT_MAX;

  K = (Kernel *) calloc(1,sizeof(Kernel));
  if (K == NULL){
    Error(MSG1,"CreateKernel");
  }
  K->val = AllocFloatArray(A->n);
  K->adj = CreateAdjRel(A->n);

  for (i=0;i<A->n;i++) {
    max.x = MAX(A->dx[i],max.x);
    max.y = MAX(A->dy[i],max.y);
    min.x = MIN(A->dx[i],min.x);
    min.y = MIN(A->dy[i],min.y);
    K->adj->dx[i] = A->dx[i];
    K->adj->dy[i] = A->dy[i];
  }

  K->xsize = max.x - min.x + 1;
  K->ysize = max.y - min.y + 1;

  return(K);
}


Kernel *CloneKernel(Kernel *K){
  Kernel *C;

  C = (Kernel *) calloc(1,sizeof(Kernel));
  if(C == NULL)
    Error(MSG1,"CloneKernel");

  C->val = AllocFloatArray(K->adj->n);
  memcpy(C->val, K->val,
	 sizeof(float)*K->adj->n);
  C->adj   = CloneAdjRel(K->adj);
  C->xsize = K->xsize;
  C->ysize = K->ysize;

  return C;
}


Kernel *NormalizeKernel(Kernel *K){
  Kernel *C = CloneKernel(K);
  float wt=0.0;
  int i;

  for(i=0; i<K->adj->n; i++)
    wt += K->val[i];
  for(i=0; i<C->adj->n; i++)
    C->val[i] /= wt;

  return C;
}


void DestroyKernel(Kernel **K)
{
  Kernel *aux;

  aux = *K;
  if(aux != NULL){
    if (aux->val != NULL)   free(aux->val);
    DestroyAdjRel(&(aux->adj));
    free(aux);
    *K = NULL;
  }
}

void WriteKernelImage(Kernel *K, char *filename)
{
  Image *img = CreateImage(K->xsize, K->ysize);
  Pixel u,v;
  int p,i;
  float max= -FLT_MAX, c;
  u.x = K->xsize/2;
  u.y = K->ysize/2;
  SetImage (img, 128);
  for (i=0; i<K->adj->n; i++) {
    max = MAX(max, fabs(K->val[i]));
  }
  if (max != 0.0)
    c = 127.0/max;
  else
    c = 0.0;
  for (i=0; i<K->adj->n; i++) {
    v.x = u.x + K->adj->dx[i];
    v.y = u.y + K->adj->dy[i];
    p = v.x + img->tbrow[v.y];
    img->val[p] = 128 + ROUND(c * K->val[i]);
  }
  WriteImage(img, filename);
  DestroyImage(&img);
}

Image *LinearFilter(Image *img, Kernel *K) /* generic kernel */
{
  Image *cimg;
  Pixel u,v,o;
  float conv;
  int p,i,q;
  Kernel *Kf;


  cimg = CreateImage(img->ncols+K->xsize-1,img->nrows+K->ysize-1);
  Kf   = FoldKernel(K);
  o.x=o.y=INT_MIN;
  for (i=0; i < Kf->adj->n; i++){ /* (o.x,o.y) is the origin of the original image with respect to the output image */
    if (Kf->adj->dx[i] > o.x)
      o.x = Kf->adj->dx[i];
    if (Kf->adj->dy[i] > o.y)
      o.y = Kf->adj->dy[i];
  }

  for (u.y=0; u.y < cimg->nrows; u.y++)
    for (u.x=0; u.x < cimg->ncols; u.x++) {
      conv = 0.0;
      p = u.x + cimg->tbrow[u.y];
      for (i=0; i < Kf->adj->n; i++){
	v.x = u.x + Kf->adj->dx[i];
	v.y = u.y + Kf->adj->dy[i];
	if (ValidPixel(img,v.x-o.x,v.y-o.y)){
	  q = v.x-o.x + img->tbrow[v.y-o.y];
	  conv += (float)img->val[q]*Kf->val[i];
	}
      }
      cimg->val[p]=(int)conv;
    }

  DestroyKernel(&Kf);
  return(cimg);
}

Image *LinearFilter2(Image *img, Kernel *K) /* generic kernel */
{
  Image *cimg;
  Pixel u;
  AdjPxl *pxl;
  float conv;
  unsigned int p, i, x, y;

  cimg = CreateImage(img->ncols, img->nrows); /* don't change img size */
  pxl = AdjPixels(img, K->adj);

  for (u.y=0; u.y < img->nrows; ++u.y)
    for (u.x=0; u.x < img->ncols; ++u.x){
      conv = 0.0;
      p = u.x + img->tbrow[u.y];
      for (i=0; i<K->adj->n; ++i) {
        x = u.x + K->adj->dx[i];
        y = u.y + K->adj->dy[i];
        if ((x >= 0) && (x < img->ncols) && (y >= 0) && (y < img->nrows))
          conv += (float)img->val[p + pxl->dp[i]] * K->val[i];
      }
      cimg->val[p] = (int)conv;
    }

  DestroyAdjPxl(&pxl);
  return(cimg);
}

Image *ShapeFilter(Image *bin, float perc)
{
  Image *skel,*msskel,*cont,*filt;
  int p,n;
  Pixel c,u;
  int r,dx,dy;
  AnnImg *aimg;
  AdjRel *A;

  /* Compute Euclidean IFT */

  cont = LabelContPixel(bin);
  aimg = Annotate(bin,NULL,cont);
  A    = Circular(1.5);
  n      = aimg->img->ncols*aimg->img->nrows;
  for(p = 0; p < n; p++)
    if (aimg->img->val[p] == 0){
      aimg->cost->val[p] = 0;
    }
  iftDilation(aimg,A);
  DestroyImage(&cont);
  DestroyAdjRel(&A);

  /* Compute MS Skeletons */

  msskel = CompMSSkel(aimg);

  /*  Compute Skeleton */

  skel = Skeleton(msskel,perc);
  DestroyImage(&msskel);

  /* Paint circles */

  filt = CreateImage(skel->ncols,skel->nrows);
  n = skel->ncols*skel->nrows;
  for (p=0; p < n; p++){
    if (skel->val[p]==1){
      c.x = p%skel->ncols;
      c.y = p/skel->ncols;
      r   = (int)sqrt(aimg->cost->val[p]);
      for(u.y = (c.y-r), dy=-r; u.y <= (c.y+r); u.y++,dy++)
	for(u.x = (c.x-r), dx=-r; u.x <= (c.x+r); u.x++,dx++)
	  if (ValidPixel(skel,u.x,u.y)&&((int)sqrt(dx*dx + dy*dy) <= r)){
	    if (aimg->cost->val[u.x + skel->tbrow[u.y]] > 0)
	      filt->val[u.x + skel->tbrow[u.y]] = 1;
	}
    }
  }

  DestroyImage(&skel);
  DeAnnotate(&aimg);

  return(filt);
}

Image *MedianFilter(Image *img, AdjRel *A)
{
  int *val,*oval,n,i,p,q;
  Pixel u,v;
  Image *med;

  val = AllocIntArray(A->n);
  med = CreateImage(img->ncols,img->nrows);
  for (u.y=0; u.y < img->nrows; u.y++)
    for (u.x=0; u.x < img->ncols; u.x++) {
      p = u.x + img->tbrow[u.y];
      n = 0;
      for (i=0; i < A->n; i++) {
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(img,v.x,v.y)){
	  q = v.x + img->tbrow[v.y];
	  val[n] = img->val[q];
	  n++;
	}
      }
      /* QuickSort is faster than BucketSort for n < 130  */
      if (n < 130) {
	qsort(val, n, sizeof(int), comp);
	oval = val;
      } else {
	oval = BucketSort(val,n,INCREASING);
      }
      med->val[p] = oval[n/2];
      if (n >= 130) free(oval);
    }
  free(val);
  return(med);
}


/**
 * Mode Filter - get the most frequent pixel value of an adjacency.
 * If all values have the same frequency, get the central pixel value.
 */
Image *ModeFilter(Image *img, AdjRel *A)
{
  int *val,n,i,p,q,*oval;
  int mode=0,fmode,m,fm,mid,fmid=0;
  Pixel u,v;
  Image *modeimg;

  val = AllocIntArray(A->n);
  modeimg = CreateImage(img->ncols,img->nrows);
  for (u.y=0; u.y < img->nrows; ++u.y)
    for (u.x=0; u.x < img->ncols; ++u.x) {
      p = u.x + img->tbrow[u.y];
      n = 0;
      for (i=0; i < A->n; ++i) {
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(img,v.x,v.y)){
	  q = v.x + img->tbrow[v.y];
	  val[n] = img->val[q];
	  n++;
	}
      }
      /* QuickSort is faster than BucketSort for n < 130  */
      if (n < 130) {
	qsort(val, n, sizeof(int), comp);
	oval = val;
      } else {
	oval = BucketSort(val,n,INCREASING);
      }
      mode  = mid  = img->val[p];
      fmode = fmid = 1;
      m     = oval[0];
      fm    = 1;
      for (i=1; i<n; ++i) {
        if (oval[i] != m) {
          if (m == mid)
            fmid = fm;
          if (fm >= fmode) {
            mode  = m;
            fmode = fm;
          }
          m  = oval[i];
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
      if (n >= 130) free(oval);

      modeimg->val[p] = fmid == fmode ? mid : mode;
    }
  free(val);
  return(modeimg);
}

Image *LaplacianBorder (Image *img, Kernel *K) {

  Image *limg = NULL;
  Image *cimg = NULL;
  Image *nimg = NULL;

  Pixel u,v;
  int p,q,i,sz,cval,Imax,s=0;

  Imax = MaximumValue(img);
  while (Imax > 255) {
    s++; Imax>>=1;
  }

  sz = FrameSize(K->adj);
  cimg = LinearFilter(img,K);
  limg = RemFrame(cimg,sz);
  DestroyImage(&cimg);
  cimg = CreateImage(limg->ncols,limg->nrows);

  for (u.y=0; u.y < limg->nrows; ++u.y)
    for (u.x=0; u.x < limg->ncols; ++u.x) {
      p = u.x + limg->tbrow[u.y];
      if (limg->val[p] >=0) {
        cval = 0;
        for (i=0; i < K->adj->n; ++i) {
	  v.x = u.x + K->adj->dx[i];
	  v.y = u.y + K->adj->dy[i];
	    if (ValidPixel(limg,v.x,v.y)){
	      q = v.x + limg->tbrow[v.y];
              if (limg->val[q] <0)
                cval = 1;
	    }
	}
        cimg->val[p] = cval;
      }
    }
  DestroyImage(&limg);
  nimg = LinearStretch(cimg,0,MaximumValue(cimg),0,Imax);
  DestroyImage(&cimg);
  return (nimg);
}


Kernel *LinearKernel(AdjRel *A)
{
  float dmax=0,*d;
  Kernel *K;
  int i;

  K = CreateKernel(A);
  d = AllocFloatArray(A->n);
  for (i=0;i<A->n;i++) {
    d[i] = sqrt(A->dx[i] * A->dx[i] + A->dy[i] * A->dy[i]);
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

Kernel *GaussianKernel(AdjRel *A, float stddev)
{
  double k,k1,d2,sigma,sigma2;
  Kernel *K;
  int i;

  d2 = 0.0;
  for (i=0;i<A->n;i++) {
    d2 = MAX((double)(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]),d2);
  }
  sigma2 = d2 / (stddev*stddev);
  sigma = sqrt(sigma2);
  k = 2.0*sigma2;
  k1 = 1.0/(sqrt(2*PI)*sigma);
  K = CreateKernel(A);
  for (i=0;i<A->n;i++) {
    d2 = (double)(A->dx[i] * A->dx[i] + A->dy[i] * A->dy[i]);
    K->val[i] = (float)(k1*exp(-d2/k)); // gaussian
  }
  return(K);

}

Kernel *LaplacianKernel(AdjRel *A, float stddev)
{

  double k,k1,d2,sigma2;
  Kernel *K;
  int i;

  d2 = 0.0;
  for (i=0;i<A->n;i++) {
    d2 = MAX((double)(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]),d2);
  }

  sigma2 = (d2/ (stddev*stddev));
  k = 2.0*sigma2;
  k1 = 1.0/sqrt(2*PI);
  K = CreateKernel(A);
  for (i=0;i<A->n;i++) {
    d2 = (double)(A->dx[i] * A->dx[i] + A->dy[i] * A->dy[i]);
    K->val[i] = (float)((d2/(2*sigma2)+1.0)* k1 * exp(-d2/k)); // laplacian
  }
  return(K);

}


/*
lamda represents the wavelength of the cosine factor.
theta represents the orientation of the normal to the parallel stripes of a Gabor function.
psi is the phase offset.
gamma is the spatial aspect ratio, and specifies the ellipticity of the support of the Gabor function.
http://en.wikipedia.org/wiki/Gabor_filter
*/

Kernel *GaborKernel(AdjRel *A, float sigma_x, float sigma_y, float theta, float lambda, float psi, float gamma)
{
  double x_theta, y_theta;
  double x_theta2, y_theta2;
  double sigma_x2, sigma_y2;
  double gamma2 = gamma*gamma;
  Kernel *K;
  int i;

  K = CreateKernel(A);

  sigma_x2 = sigma_x*sigma_x;
  sigma_y2 = sigma_y*sigma_y;

  for (i=0;i<A->n;i++) {
    x_theta = A->dx[i]*cos(theta)+A->dy[i]*sin(theta);
    x_theta2 = x_theta * x_theta;
    y_theta = -A->dx[i]*cos(theta)+A->dy[i]*sin(theta);
    y_theta2 = y_theta * y_theta;
    K->val[i] = (float)(exp(-.5*((x_theta2/sigma_x2)+(gamma2*y_theta2/sigma_y2))) * cos(2*PI*x_theta/lambda+ psi));
  }
  return(K);

}

/* Reflect kernel around x and y */

Kernel *FoldKernel(Kernel *K)
{
  Kernel *Kf;
  int i,j;

  Kf = CreateKernel(K->adj);
  for (i=0,j=Kf->adj->n-1; i < Kf->adj->n; i++,j--){
    Kf->val[i]     = K->val[j];
    Kf->adj->dx[i] = -K->adj->dx[j];
    Kf->adj->dy[i] = -K->adj->dy[j];
  }
  return(Kf);
}

Image *ImageConvolution(Image *img, Kernel *K)
{
  Image *cimg;
  Pixel u,v;
  float conv;
  int p,i,q;
  Kernel *Kf;

  cimg = CreateImage(img->ncols+K->xsize-1,img->nrows+K->ysize-1);
  Kf   = FoldKernel(K);

  for (u.y=0; u.y < cimg->nrows; u.y++)
    for (u.x=0; u.x < cimg->ncols; u.x++) {
      conv = 0.0;
      p = u.x + cimg->tbrow[u.y];
      for (i=0; i < Kf->adj->n; i++){
	v.x = u.x + Kf->adj->dx[i];
	v.y = u.y + Kf->adj->dy[i];
	if (ValidPixel(img,v.x,v.y)){
	  q = v.x + img->tbrow[v.y];
	  conv += (float)img->val[q]*Kf->val[i];
	}
      }
      cimg->val[p]=(int)conv;
    }

  DestroyKernel(&Kf);
  return(cimg);

}

real *RealConvolution2D(real *value, Kernel *K, int ncols, int nrows)
{
  real *cimg, conv;
  Pixel u,v;
  int p,i,q;
  Kernel *Kf;
  Image img;
  img.ncols = ncols;
  img.nrows = nrows;

  cimg = AllocRealArray(ncols * nrows);
  Kf   = FoldKernel(K);

  for (u.y=0; u.y < nrows; u.y++)
    for (u.x=0; u.x < ncols; u.x++) {
      conv = 0.0;
      p = u.x + u.y * ncols;
      for (i=0; i < Kf->adj->n; i++){
	v.x = u.x + Kf->adj->dx[i];
	v.y = u.y + Kf->adj->dy[i];
	if (ValidPixel(&img,v.x,v.y)){
	  q = v.x + v.y * ncols;
	  conv += value[q] * Kf->val[i];
	}
      }
      cimg[p] = conv;
    }

  DestroyKernel(&Kf);
  return(cimg);

}

Image *SobelFilter(Image *img)
{

  Image *gradx=NULL,*grady=NULL,*grad=NULL;
  Kernel *Kx,*Ky;


  Ky = MakeKernel("3,3,-1.0,-2.0,-1.0,0.0,0.0,0.0,1.0,2.0,1.0");
  Kx = MakeKernel("3,3,-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0");
  gradx = LinearFilter2(img,Kx);
  grady = LinearFilter2(img,Ky);
  grad  = ImageMagnitude(gradx,grady);

  DestroyImage(&gradx);
  DestroyImage(&grady);
  DestroyKernel(&Kx);
  DestroyKernel(&Ky);

  return(grad);
}


