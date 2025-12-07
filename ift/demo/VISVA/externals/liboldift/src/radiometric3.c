#include "radiometric3.h"
#include "matrix.h"

Curve *Histogram3(Scene *scn)
{
  int i, n, nbins;
  Curve *hist = NULL;

  nbins = MaximumValue3(scn)+1;
  hist  = CreateCurve(nbins);
  n = scn->xsize * scn->ysize * scn->zsize;
  for (i = 0; i < n; i++)
    hist->Y[scn->data[i]]++;
  for (i = 0; i < nbins; i++)
    hist->X[i] = i;

  return (hist);
}


#ifndef _MSC_VER

Curve *RegionHistogram3( Scene *scn, Subgraph *sg, int label ) {
  int i, n, maxbin, minbin;
  Curve *hist = NULL;

  n = scn->xsize * scn->ysize * scn->zsize;
  maxbin = -1;
  minbin = INT_MAX;
  for ( i = 0; i < n; i++ ) {
    if ( sg->node[ i ].label == label ) {
      if ( maxbin < scn->data[ i ] ) maxbin = scn->data[ i ];
      if ( minbin > scn->data[ i ] ) minbin = scn->data[ i ];
    } 
  } 


  hist  = CreateCurve( maxbin - minbin + 1 );
  for ( i = 0; i < n; i++ ) {
    if ( sg->node[ i ].label == label ) hist->Y[ scn->data[ i ] - minbin ]++;
  }
  for ( i = 0; i < maxbin - minbin + 1; i++ ) hist->X[ i ] = i + minbin;

  return ( hist );
}

Curve *LabelSubGraphHistogram( Scene *scn, LNode *ln ) {
  Set *aux;
  Curve *hist = NULL;
  int minval = INT_MAX, maxval = INT_MIN;
  int i, n = 0;
  
  for ( aux = ln->voxels; aux != NULL; aux = aux->next ) {
    if ( minval > scn->data[ aux->elem ] ) minval = scn->data[ aux->elem ];
    if ( maxval < scn->data[ aux->elem ] ) maxval = scn->data[ aux->elem ];
    n++;
  }

  hist = CreateCurve( maxval - minval + 1 );
  for ( i = minval; i <= maxval; i++ ) {
    hist->X[ i - minval ] = i;
  }
  for ( aux = ln->voxels; aux != NULL; aux = aux->next ) {
    hist->Y[ scn->data[ aux->elem ] - minval ]++;
  }
  for ( i = minval; i <= maxval; i++ ) {
    hist->Y[ i - minval ] /= n;
  }
  return ( hist );
}

Curve *LabelSubGraphSimpleHistogram( Scene *scn, LNode *ln, float sampling ) {
  Set *aux;
  Curve *hist = NULL;
  float minval = FLT_MAX, maxval = -FLT_MAX;
  int i, n = 0, I;
  
  for ( aux = ln->voxels; aux != NULL; aux = aux->next ) {
    if ( minval > scn->data[ aux->elem ] ) minval = ( float ) scn->data[ aux->elem ];
    if ( maxval < scn->data[ aux->elem ] ) maxval = ( float ) scn->data[ aux->elem ];
    n++;
  }
  
  I = MaximumValue3( scn ) * sampling;
  if ( I == 0 ) I = 1;
  minval = ( float ) ( ( int ) minval / I );
  maxval = ( float ) ( ( int ) maxval / I );
  hist = CreateCurve( maxval - minval + 1 );
  //printf( "max = %f, min = %f, n = %d\n", maxval, minval, hist->n );
  for ( i = minval; i <= maxval; i++ ) {
    hist->X[ ( int ) ( i - minval ) ] = i;
  }
  for ( aux = ln->voxels; aux != NULL; aux = aux->next ) {
    hist->Y[ ( int ) ( scn->data[ aux->elem ] / I - minval ) ]++;
  }
  for ( i = minval; i <= maxval; i++ ) {
    hist->Y[ ( int ) ( i - minval ) ] /= n;
  }
  return ( hist );
}

#endif


Curve *NormHistogram3(Scene *scn)
{
  int i, sum;
  Curve *nhist;

  nhist = Histogram3(scn);
  sum = scn->xsize * scn->ysize * scn->zsize;
  for (i = 0; i < nhist->n; i++){
    nhist->Y[i] /= sum;
    nhist->X[i]=i;
  }

  return (nhist);
}

Curve *AccHistogram3(Scene *scn)
{
  int i;
  Curve *ahist;

  ahist = Histogram3(scn);
  for (i = 1; i < ahist->n; i++){
    ahist->Y[i] += ahist->Y[i-1];
    ahist->X[i] = i;
  }

  return (ahist);
}

Curve *NormAccHistogram3(Scene *scn)
{
  int i;
  Curve *ahist;

  ahist = NormHistogram3(scn);
  for (i = 1; i < ahist->n; i++){
    ahist->Y[i] = ahist->Y[i-1] + ahist->Y[i];
    ahist->X[i] = i;
  }
  return (ahist);
}

Curve *HistogramMask3(Scene *scn,
		      Scene *mask){
  int i,p,n,nbins,Imax;
  Curve *hist = NULL;

  n = scn->xsize * scn->ysize * scn->zsize;

  Imax = MaximumValueMask3(scn, mask);

  nbins = Imax+1;
  hist  = CreateCurve(nbins);

  for(p=0; p<n; p++)
    if(mask->data[p]>0)
      hist->Y[scn->data[p]]++;
  for(i=0; i<nbins; i++)
    hist->X[i] = i;

  return (hist);
}

Curve *NormHistogramMask3(Scene *scn,
			  Scene *mask){
  int i, sum;
  Curve *nhist;

  nhist = HistogramMask3(scn, mask);

  sum = 0;
  for(i=0; i<nhist->n; i++)
    sum += ROUND(nhist->Y[i]);
  for(i=0; i<nhist->n; i++){
    nhist->Y[i] /= (double)sum;
    nhist->X[i]=i;
  }

  return (nhist);
}

Curve *AccHistogramMask3(Scene *scn, Scene *mask)
{
  int i;
  Curve *ahist;

  ahist = HistogramMask3(scn, mask);
  for (i = 1; i < ahist->n; i++){
    ahist->Y[i] += ahist->Y[i-1];
    ahist->X[i] = i;
  }

  return (ahist);
}

Curve *NormAccHistogramMask3(Scene *scn, Scene *mask)
{
  int i;
  Curve *ahist;

  ahist = NormHistogramMask3(scn,mask);
 
  for (i = 1; i < ahist->n; i++){
    ahist->Y[i] = ahist->Y[i-1] + ahist->Y[i];
    ahist->X[i] = i;
  }
  return (ahist);
}

float HistogramMatching( Curve *hp, Curve *hq ) {
  float matching = 0.0;
  int i = 0, j = 0;
  while ( ( i < hp->n ) && ( j < hq->n ) ) {
    if ( hp->X[ i ] == hq->X[ j ] ) {
      matching += MIN( hp->Y[ i ], hq->Y[ j ] );
      i++;
      j++;
    }
    else if ( hp->X[ i ] > hq->X[ j ] ) j++;
    else i++;
  }
  //if ( matching > 0.5 ) printf( "matching = %f\n", matching );
  return matching;
}

Scene *GaussStretch3(Scene *scn, float mean, float stdev)
{
  float *gauss=NULL,sq,var2;
  int i,Imax,n;
  Scene *gscn=NULL;

  Imax  = MaximumValue3(scn);
  gauss = AllocFloatArray(Imax+1);
  var2  = 2*stdev*stdev;
  for (i=0; i < Imax+1; i++){
    sq  = ((float)i-mean)*((float)i-mean);
    gauss[i]=(float)(Imax*exp(-sq/var2));
  }
  n     = scn->xsize*scn->ysize*scn->zsize;
  gscn     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  gscn->dx = scn->dx;
  gscn->dy = scn->dy;
  gscn->dz = scn->dz;

  for (i=0; i < n; i++){
    gscn->data[i] = (int)(gauss[scn->data[i]]);
  }
  gscn->dx = scn->dx;
  gscn->dy = scn->dy;
  gscn->dz = scn->dz;

  free(gauss);
  return(gscn);  
}

Scene *LinearStretch3(Scene *scn, int f1, int f2, int g1, int g2)
{
  Scene *sscn = NULL;
  int p, n;
  float a;

  sscn = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  sscn->dx = scn->dx;
  sscn->dy = scn->dy;
  sscn->dz = scn->dz;

  n = scn->xsize * scn->ysize * scn->zsize;

  if (f1 != f2)
    a = (float)(g2-g1)/(float)(f2-f1);
  else
    a = INT_MAX;

  for (p = 0; p < n; p++) {
    if (scn->data[p] < f1)
      sscn->data[p] = g1;
    else 
      if (scn->data[p] > f2)
	sscn->data[p] = g2;
      else{
	if (a != INT_MAX)
	  sscn->data[p] = (int)(a*(scn->data[p]-f1)+g1);
	else 
	  sscn->data[p] = g2;
      }
  }

  return (sscn);
}


void   LinearStretchinplace3(Scene *scn, 
			     int f1, int f2, 
			     int g1, int g2){
  int p,n;
  float a;
  
  n = scn->xsize*scn->ysize*scn->zsize;
  
  if(f1 != f2)
    a = (float)(g2-g1)/(float)(f2-f1);
  else
    a = INT_MAX;
  
  for(p=0; p<n; p++){
    if(scn->data[p] < f1)
      scn->data[p] = g1;
    else 
      if(scn->data[p] > f2)
	scn->data[p] = g2;
      else{
	if(a != INT_MAX)
	  scn->data[p] = (int)(a*(scn->data[p]-f1)+g1);
	else 
	  scn->data[p] = g2;
      }
  }
}


Curve *SceneLabels (Scene *scn) {

  int i,j=0,n=0;
  Curve *c,*out;

  c = Histogram3(scn);
  for (i=1;i<c->n;i++) {
    if (c->Y[i] > 0) {
      n++;
    }
  }
  out = CreateCurve(n);
  for (i=1;i<c->n;i++) {
    if (c->Y[i] > 0) {
      out->X[j] = c->X[i];
      out->Y[j] = c->Y[i];
      j++;
    }
  }
  DestroyCurve(&c);
  return(out);

}

// Equalize scene from 0 to Imax, using the voxel sorting approach. 
Scene *Equalize3(Scene *scn, int Imax)
{
  int     p,n,val,nelems,nelems_per_value;
  Scene  *escn=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  GQueue *Q=NULL;

  n = scn->xsize*scn->ysize*scn->zsize;
  Q = CreateGQueue(MaximumValue3(scn)+1,n,scn->data);

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
    escn->data[p]=val;
    nelems++;
  }
  DestroyGQueue(&Q);
  return(escn);
}

// Equalize scene from 0 to Imax, using the voxel sorting approach. 
Scene *EqualizeMask3(Scene *scn, Scene *mask, int Imax)
{
  int     p,n,size,val,nelems,nelems_per_value;
  Scene  *escn=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  GQueue *Q=NULL;

  n = scn->xsize*scn->ysize*scn->zsize;
  Q = CreateGQueue(MaximumValue3(scn)+1,n,scn->data);
  
  size = 0;
  for (p=0; p < n; p++) {
    if ( mask->data[ p ] == 1 ) {
      InsertGQueue(&Q,p);
      size++;
    }
  }

  nelems_per_value = (int)((float)size/(float)(Imax+1));

  nelems=1; val=0;
  while (!EmptyGQueue(Q)){
    p = RemoveGQueue(Q);
    if (nelems > nelems_per_value){   
      val    = MIN(val+1,Imax);
      nelems = 0;
    }
    escn->data[p]=val;
    nelems++;
  }
  DestroyGQueue(&Q);
  return(escn);
}

// Match histograms using the voxel sorting approach

Scene *MatchHistogram3(Scene *scn, Scene *des)
{
  int     p1,p2,i,n1,n2,Imax;
  Scene  *mscn=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  int     nelems1,nelems2,nelems1_per_value,nelems2_per_value;
  GQueue *Q1,*Q2;

  n1     = scn->xsize*scn->ysize*scn->zsize;
  n2     = des->xsize*des->ysize*des->zsize;
  Imax   = MAX(MaximumValue3(scn),MaximumValue3(des));

  Q1 = CreateGQueue(Imax+1,n1,scn->data);
  Q2 = CreateGQueue(Imax+1,n2,des->data);

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
      mscn->data[p1] = des->data[p2];	
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
	  mscn->data[p1] = des->data[p2];	
	  i++;
	  nelems1++;
	}
      }
    }
  }
  while (!EmptyGQueue(Q1)&& 
	 (i < n1)){
    p1 = RemoveGQueue(Q1);
    mscn->data[p1] = des->data[p2];	
    i++;
  }

  DestroyGQueue(&Q1);
  DestroyGQueue(&Q2);
  return(mscn);
}

// Match histograms with masks, using the voxel sorting approach

Scene *MatchHistogramMasks3(Scene *scn, Scene *scn_mask, \
			    Scene *des, Scene *des_mask)
{
  int     p1,p2,i,n1,n2,mn1,mn2,mImin,mImax;
  Scene  *mscn=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  int     nelems1,nelems2,nelems1_per_value,nelems2_per_value;
  GQueue *Q1,*Q2;

  n1     = scn->xsize*scn->ysize*scn->zsize;
  mImax  = INT_MIN; mImin = INT_MAX;
  mn1    = 0;
  for (p1=0; p1 < n1; p1++) {
    if (scn_mask->data[p1]==1){
      if (scn->data[p1]>mImax)
	mImax = scn->data[p1];
      if (scn->data[p1]<mImin)
	mImin = scn->data[p1];
      mn1++;
    }
  }

  n2     = des->xsize*des->ysize*des->zsize;
  mn2    = 0;
  for (p2=0; p2 < n2; p2++) {
    if (des_mask->data[p2]==1){
      if (des->data[p2]>mImax)
	mImax = des->data[p2];
      if (des->data[p2]<mImin)
	mImin = des->data[p2];
      mn2++;
    }
  }

  Q1 = CreateGQueue(mImax+1,n1,scn->data);
  Q2 = CreateGQueue(mImax+1,n2,des->data);

  for (p1=0; p1 < n1; p1++) 
    if (scn_mask->data[p1]==1)
      InsertGQueue(&Q1,p1);

  for (p2=0; p2 < n2; p2++)
    if (des_mask->data[p2]==1)
      InsertGQueue(&Q2,p2);

  nelems1_per_value = (int)((float)mn1/(float)(mImax-mImin+1));
  nelems2_per_value = (int)((float)mn2/(float)(mImax-mImin+1));

  i=0;
  while ((i < n1)&&(!EmptyGQueue(Q1))&&(!EmptyGQueue(Q2)))  {
    if (scn_mask->data[i]==1){
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
	mscn->data[p1] = des->data[p2];	
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
	    mscn->data[p1] = des->data[p2];	
	    i++;
	    nelems1++;
	  }
	}
      }
    }else{
      i++;
    }
  }
  while (!EmptyGQueue(Q1)&& 
	 (i < n1)){
    p1 = RemoveGQueue(Q1);
    mscn->data[p1] = des->data[p2];	
    i++;
  }
  
  DestroyGQueue(&Q1);
  DestroyGQueue(&Q2);
  return(mscn);
}


Scene *TradEqualize3(Scene *scn)
{
  int p,n,Imax;
  Scene *escn=NULL;
  Curve *nhist;

  nhist = NormAccHistogram3(scn);
  escn  = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  escn->dx = scn->dx;
  escn->dy = scn->dy;
  escn->dz = scn->dz;
  n     = scn->xsize*scn->ysize*scn->zsize;
  Imax  = MaximumValue3(scn);

  if (Imax < 255) /* 8 bits */
    Imax = 255;
  else
    if (Imax < 4095) /* 12 bits */
      Imax = 4095;

  for (p=0; p < n; p++)
    escn->data[p] = (int)(Imax*nhist->Y[scn->data[p]]);

  DestroyCurve(&nhist);

  return(escn);
}
Scene *MaskEqualize3(Scene *scn,Scene *mask)
{
  int p,n,Imax;
  Scene *escn=NULL;
  Curve *nhist;

  nhist = NormAccHistogramMask3(scn,mask);
  escn  = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  escn->dx = scn->dx;
  escn->dy = scn->dy;
  escn->dz = scn->dz;
  n     = scn->xsize*scn->ysize*scn->zsize;
  Imax  = MaximumValueMask3(scn,mask);

  if (Imax < 255) /* 8 bits */
    Imax = 255;
  else
    if (Imax < 4095) /* 12 bits */
      Imax = 4095;

  for (p=0; p < n; p++){
  	if(mask->data[p]!=0)
    escn->data[p] = (int)(Imax*nhist->Y[scn->data[p]]);
  }
  DestroyCurve(&nhist);

  return(escn);
}

Scene *TradMatchHistogram3(Scene *scn, Scene *des)
{
  int   start,end,pos=0,p,n;
  Scene *mscn=NULL;
  Curve *nhist1=NULL,*nhist2=NULL;
  double val;
  bool found;

  nhist1 = NormAccHistogram3(scn);
  nhist2 = NormAccHistogram3(des);
  n      = scn->zsize*scn->ysize*scn->xsize;
  mscn   = CreateScene(scn->xsize,scn->ysize,scn->zsize);

  for (p=0; p < n; p++) {
    val   = nhist1->Y[scn->data[p]];
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
      mscn->data[p]=pos;
    else
      if (fabs(val-nhist2->Y[start])<fabs(val-nhist2->Y[end]))
	mscn->data[p]=start;
      else
	mscn->data[p]=end;
  }

  DestroyCurve(&nhist1);
  DestroyCurve(&nhist2);

  return(mscn);
}

Matrix * JoinHistogram(Scene *scn1, Scene *scn2){

	int n,L1,L2,l1,l2,i;
	Matrix* joinHist;
	L1=scn1->maxval;
	L2=scn2->maxval;

	joinHist=CreateMatrix(L1+1,L2+1);
	
	n=joinHist->nrows*joinHist->ncols;
	for(i=0;i<n;i++)
		joinHist->val[i]=0;
		
	n=scn1->xsize*scn1->ysize*scn1->zsize;	
	for(i=0;i<n;i++){
		l1=scn1->data[i];
		l2=scn2->data[i];
		joinHist->val[joinHist->tbrow[l2]+l1]=(joinHist->val[joinHist->tbrow[l2]+l1])+1;
	}
	
	return joinHist;

}

Matrix *NormJoinHistogram(Scene *scn1, Scene *scn2){

	Matrix *h=JoinHistogram(scn1,scn2);
	Matrix *norm=CreateMatrix(h->ncols,h->nrows);
	double sum=(double)(scn1->xsize*scn1->ysize*scn1->zsize);
	int n=h->ncols*h->nrows;
	int i;
	for(i=0;i<n;i++){
		norm->val[i]=h->val[i]/sum;
	}
	DestroyMatrix(&h);
	return norm;
}


