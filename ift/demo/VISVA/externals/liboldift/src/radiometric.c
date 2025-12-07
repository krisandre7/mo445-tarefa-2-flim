#include "radiometric.h"

Curve *Histogram(Image *img)
{
  int i,p,n,nbins;
  Curve *hist=NULL;

  nbins = MaximumValue(img)+1;
  hist  = CreateCurve(nbins);
  n     = img->ncols*img->nrows;
  for (p=0; p < n; p++)
    hist->Y[img->val[p]]++;
  for (i=0; i < nbins; i++) 
    hist->X[i] = i;

  return(hist);
}

Curve *NormHistogram(Image *img) 
{
  int i,sum;
  Curve *hist=NULL,*nhist=NULL;

  hist  = Histogram(img);
  sum   = img->ncols*img->nrows;
  nhist = CreateCurve(hist->n);
  for (i=0; i < nhist->n;i++){
    nhist->Y[i] = hist->Y[i]/sum;
    nhist->X[i] = hist->X[i];
  }

  DestroyCurve(&hist);

  return(nhist);
}

Image *Probability(Image *img) 
{
  int p,n;
  Image *prob;
  Curve *nhist;

  n     = img->ncols*img->nrows;
  prob  = CreateImage(img->ncols,img->nrows);
  nhist = NormHistogram(img);
  for (p=0; p < n; p++)
      prob->val[p] = (int)(100*nhist->Y[img->val[p]]);

  DestroyCurve(&nhist);
  return(prob);
}

Curve *AccHistogram(Image *img)
{
  int i;
  Curve *hist=NULL,*ahist=NULL;

  hist = Histogram(img);
  ahist = CreateCurve(hist->n);
  ahist->X[0] = 0;
  ahist->Y[0] = hist->Y[0];
  for (i=1; i < ahist->n; i++){
    ahist->X[i] = i;
    ahist->Y[i] = ahist->Y[i-1] + hist->Y[i];
  }

  DestroyCurve(&hist);

  return(ahist);
}

Curve *NormAccHistogram(Image *img)
{
  int i,sum;
  Curve *ahist=NULL,*nahist=NULL;

  ahist  = AccHistogram(img);
  sum    = img->ncols*img->nrows;
  nahist = CreateCurve(ahist->n);
  for (i=0; i < nahist->n;i++){
    nahist->Y[i] = ahist->Y[i]/sum;
    nahist->X[i] = ahist->X[i];
  }
  DestroyCurve(&ahist);

  return(nahist);
}


Curve *NormalizeHistogram(Curve *hist){
  Curve *nhist;
  double sum;
  int i;
  
  nhist = CopyCurve(hist);

  sum = 0.0;
  for(i=0; i<nhist->n; i++)
    sum += nhist->Y[i];
  for(i=0; i<nhist->n; i++)
    nhist->Y[i] /= sum;
  
  return (nhist);
}

// Equalize image from 0 to Imax, using the pixel sorting approach. 

Image *Equalize(Image *img, int Imax)
{
  int     p,n,val,nelems,nelems_per_value;
  Image  *eimg=CreateImage(img->ncols,img->nrows);
  GQueue *Q=NULL;

  n = img->ncols*img->nrows;
  Q = CreateGQueue(MaximumValue(img)+1,n,img->val);

  for (p=0; p < n; p++) 
    InsertGQueue(&Q,p);

  nelems_per_value = (int)((float)n/(float)(Imax+1));

  nelems=1; val=0;
  while (!EmptyGQueue(Q)){
    p = RemoveGQueue(Q);
    if (nelems > nelems_per_value){      
      val    = MIN(val+1,Imax);
      nelems = 0;
    }
    eimg->val[p]=val;
    nelems++;
  }
  DestroyGQueue(&Q);
  return(eimg);
}

// Match histograms using the pixel sorting approach

Image *MatchHistogram(Image *img, Image *des)
{
  int     p1,p2,i,n1,n2,Imax;
  Image  *mimg=CreateImage(img->ncols,img->nrows);
  int     nelems1,nelems2,nelems1_per_value,nelems2_per_value;
  GQueue *Q1,*Q2;

  n1     = img->nrows*img->ncols;
  n2     = des->nrows*des->ncols;
  Imax   = MAX(MaximumValue(img),MaximumValue(des));

  Q1 = CreateGQueue(Imax+1,n1,img->val);
  Q2 = CreateGQueue(Imax+1,n2,des->val);

  for (p1=0; p1 < n1; p1++) 
    InsertGQueue(&Q1,p1);

  for (p2=0; p2 < n2; p2++) 
    InsertGQueue(&Q2,p2);

  nelems1_per_value = (int)((float)n1/(float)(Imax+1));
  nelems2_per_value = (int)((float)n2/(float)(Imax+1));

  i=0;
  while ((i < n1)&&(!EmptyGQueue(Q1))&&(!EmptyGQueue(Q2)))  {
    nelems1=nelems2=1;
    while ((nelems1 <= nelems1_per_value)&&
	   (nelems2 <= nelems2_per_value)&& 
	   (i < n1)){    
      if (!EmptyGQueue(Q1)){
	p1 = RemoveGQueue(Q1);
	nelems1++;
      }else{
	break;
      }
      if (!EmptyGQueue(Q2)){
	p2 = RemoveGQueue(Q2);
	nelems2++;
      }else{
	break;
      }
      mimg->val[p1] = des->val[p2];	
      i++;
    }
    if (nelems1>nelems1_per_value){
      while ((!EmptyGQueue(Q2))&& 
	     (nelems2 <= nelems2_per_value)){
	p2 = RemoveGQueue(Q2);
	nelems2++;
      }
    }else{
      if (nelems2>nelems2_per_value){
	while ((!EmptyGQueue(Q1))&& 
	       (nelems1 <= nelems1_per_value)&& 
	       (i < n1)){
	  p1 = RemoveGQueue(Q1);
	  mimg->val[p1] = des->val[p2];	
	  i++;
	  nelems1++;
	}
      }
    }
  }
  while (!EmptyGQueue(Q1)&& 
	 (i < n1)){
    p1 = RemoveGQueue(Q1);
    mimg->val[p1] = des->val[p2];	
    i++;
  }

  DestroyGQueue(&Q1);
  DestroyGQueue(&Q2);

  return(mimg);
}

Image *LinearStretch(Image *img, int f1, int f2, int g1, int g2)
{
  Image *simg=NULL;
  int p,n;
  float a;

  simg = CreateImage(img->ncols,img->nrows);
  n    = img->ncols*img->nrows;
  if (f1 != f2) 
    a = (float)(g2-g1)/(float)(f2-f1);
  else
    a = INT_MAX;

  for (p=0; p < n; p++){
    if (img->val[p] < f1)
      simg->val[p] = g1;
    else 
      if (img->val[p] > f2)
	simg->val[p] = g2;
      else {
	if (a != INT_MAX)	  
	  simg->val[p] = (int)(a*(img->val[p]-f1)+g1);
	else{
	  simg->val[p] = g2;
	}   
      }
  }
  return(simg);
}

Image *GaussStretch(Image *img, float mean, float stdev)
{
  float *gauss=NULL,sq,var2;
  int i,Imax,n;
  Image *gimg=NULL;

  Imax  = MaximumValue(img);
  gauss = AllocFloatArray(Imax+1);
  var2  = 2*stdev*stdev;
  for (i=0; i < Imax+1; i++){
    sq  = ((float)i-mean)*((float)i-mean);
    gauss[i]=(float)(Imax*exp(-sq/var2));
  }
  n     = img->ncols*img->nrows;
  gimg     = CreateImage(img->ncols,img->nrows);
  for (i=0; i < n; i++){
    gimg->val[i] = (int)(gauss[img->val[i]]);
  }
  free(gauss);
  return(gimg);  
}


void LinearStretchinplace(Image *img, 
			  int f1, int f2, 
			  int g1, int g2){
  int p,n;
  float a;

  n    = img->ncols*img->nrows;
  if (f1 != f2) 
    a = (float)(g2-g1)/(float)(f2-f1);
  else
    a = INT_MAX;

  for (p=0; p < n; p++){
    if (img->val[p] < f1)
      img->val[p] = g1;
    else 
      if (img->val[p] > f2)
	img->val[p] = g2;
      else {
	if (a != INT_MAX)	  
	  img->val[p] = (int)(a*(img->val[p]-f1)+g1);
	else{
	  img->val[p] = g2;
	}   
      }
  }
}

// Traditional equalization

Image *TradEqualize(Image *img)
{
  int p,n,Imax;
  Image *eimg=NULL;
  Curve *nhist;

  nhist = NormAccHistogram(img);
  eimg  = CreateImage(img->ncols,img->nrows);
  n     = img->ncols*img->nrows;
  Imax  = MaximumValue(img);

  if (Imax < 255) /* 8 bits */
    Imax = 255;
  else
    if (Imax < 4095) /* 12 bits */
      Imax = 4095;

  for (p=0; p < n; p++)
    eimg->val[p] = (int)(Imax*nhist->Y[img->val[p]]);

  DestroyCurve(&nhist);

  return(eimg);
}

// Traditional Histogram Matching

Image *TradMatchHistogram(Image *img, Image *des)
{
  int   start,end,pos=0,p,n;
  Image *mimg=NULL;
  Curve *nhist1=NULL,*nhist2=NULL;
  double val;
  bool found;

  nhist1 = NormAccHistogram(img);
  nhist2 = NormAccHistogram(des);
  n      = img->nrows*img->ncols;
  mimg   = CreateImage(img->ncols,img->nrows);

  for (p=0; p < n; p++) {
    val   = nhist1->Y[img->val[p]];
    start = 0;
    end   = nhist2->n-1;
    found = false;
    while ((start <= end)&&(!found)){
      pos = (start+end)/2;
      if (val < nhist2->Y[pos])
	end = pos-1;
      else
	if (val > nhist2->Y[pos])
	  start = pos+1;
	else
	  found = true;
    }
    if (found)
      mimg->val[p]=pos;
    else
      if (fabs(val-nhist2->Y[start])<fabs(val-nhist2->Y[end]))
	mimg->val[p]=start;
      else
	mimg->val[p]=end;
  }

  DestroyCurve(&nhist1);
  DestroyCurve(&nhist2);

  return(mimg);
}




