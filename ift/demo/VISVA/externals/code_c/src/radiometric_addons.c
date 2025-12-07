
#include "radiometric_addons.h"



Curve *XClipHistogram(Curve *hist,
		      int lower, int higher){
  Curve *chist;
  int i;

  chist = CopyCurve(hist);

  if(lower>chist->n)
    lower = chist->n;
  if(higher<-1)
    higher = -1;

  for(i=0; i<lower; i++)
    chist->Y[i] = 0.0;
  for(i=chist->n-1; i>higher; i--)
    chist->Y[i] = 0.0;

  return chist;
}


Curve *ClipHistogram(Curve *hist, 
		     real left, real right,
		     real bottom, real top){
  double Cmax = CurveMaximum(hist);
  Curve *chist;
  int i,l,r;

  chist = CopyCurve(hist);
  for(i=0; i<chist->n; i++){
    if(chist->Y[i]>top*Cmax)
      chist->Y[i] = top*Cmax;

    chist->Y[i] -= bottom*Cmax;
    if(chist->Y[i]<0.0)
      chist->Y[i] = 0.0;
  }

  l = ROUND(chist->n*left);
  for(i=0; i<l; i++)
    chist->Y[i] = 0.0;

  r = ROUND(chist->n*right);
  for(i=r; i<chist->n; i++)
    chist->Y[i] = 0.0;

  return (chist);
}


real GaussianSimilarity(real a, real b, real stdev){
  real d = (a-b)/stdev;
  return (real)expf(-d*d);
}

real LinearSimilarity(real a, real b, real dmax){
  real d = (a>b)?(a-b):(b-a);
  return (dmax-d)/dmax;
}


int  HistogramMedianValue(Curve *hist){
  int count,n,i,j;
  real median;

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
      median = (real)j;
    }
    else
      median = (real)i;
  }
  else{
    if(count==n/2){
      j = i+1;
      while(ROUND(hist->Y[j])==0)
	j++;
      median = (real)(i+j)/2.0; 
    }
    else
      median = (real)i;
  }

  return ROUND(median);
}



int    AreaPercentageLowerThreshold(Curve *hist,
				    float perc){
  Curve *nhist;
  double sum=0.0,ratio;
  int i=0;

  ratio = perc/100.0;
  nhist = NormalizeHistogram(hist);
  while(sum<ratio && i<nhist->n){
    sum += nhist->Y[i];
    i++;
  }
  i = MAX(0, i-1);
  DestroyCurve(&nhist);

  return i;
}


int    AreaPercentageHigherThreshold(Curve *hist,
				     float perc){
  Curve *nhist;
  double sum=0.0,ratio;
  int i;

  ratio = 1.0 - perc/100.0;
  nhist = NormalizeHistogram(hist);
  i = nhist->n-1;
  while(sum<ratio && i>=0){
    sum += nhist->Y[i];
    i--;
  }
  i = MIN(nhist->n-1, i+1);
  DestroyCurve(&nhist);

  return i;
}


int    OtsuHistogramThreshold(Curve *hist){
  Curve *nhist=NormalizeHistogram(hist);
  double p1,p2,m1,m2,s1,s2,J,Jmax=-1.0;
  int i,T,Topt=0,Imax=nhist->n-1;

  for (T=1; T < Imax; T++){
    p1 = 0.0;
    for (i=0; i <= T; i++) 
      p1 += nhist->Y[i];
    p2 = 1.0 - p1;
    if ((p1 > 0.0)&&(p2 > 0.0)){
      m1 = 0.0;
      for (i=0; i <= T; i++) 
	m1 += nhist->Y[i]*i;
      m1 /= p1;
      m2 = 0.0;
      for (i=T+1; i <= Imax; i++) 
	m2 += nhist->Y[i]*i;
      m2 /= p2;
      s1 = 0.0;
      for (i=0; i <= T; i++) 
	s1 += nhist->Y[i]*(i-m1)*(i-m1);
      s1 /= p1;
      s2 = 0.0;
      for (i=T+1; i <= Imax; i++) 
	s2 += nhist->Y[i]*(i-m2)*(i-m2);
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

  DestroyCurve(&nhist);
  return(Topt);
}


//type: 0 - Linear, 1 - Gaussian.
int    NcutHistogramThreshold(Curve *hist, int type){
  int T,Topt=0,Imax=hist->n-1;
  double cutie,associ,assoce,ncut,Copt,hi,hj,ht,nw;
  real (*similarity)(real, real, real);
  real param,w;
  int i,j,first,last;
  Curve *F;

  for(i=0; i<=Imax; i++)
    if(hist->Y[i]>0.0)
      break;
  first = i;
  for(i=Imax; i>=0; i--)
    if(hist->Y[i]>0.0)
      break;
  last = i;

  if(type==0){
    param = (real)Imax;
    similarity = LinearSimilarity;
  }
  else{
    param = (real)0.15*(last-first+1);
    similarity = GaussianSimilarity;
  }

  F = CreateCurve(last-first);

  cutie  = 0.0;
  associ = 0.0;
  assoce = 0.0;
  for(i=first; i<=last; i++){
    hi = hist->Y[i];
    for(j=first; j<i; j++){
      hj = hist->Y[j];
      w = (*similarity)((real)i,(real)j,param);
      assoce += hi*hj*w;
    }
    w = (*similarity)((real)i,(real)i,param);
    assoce += (hi*(hi-1.0)*w)/2.0;
  }

  Copt = DBL_MAX;
  for(T=first; T<last; T++){

    ht = hist->Y[T];
    for(i=T-1; i>=first; i--){
      hi = hist->Y[i];
      w = (*similarity)((real)i,(real)T,param);
      nw = hi*ht*w;
      cutie  -= nw;
      associ += nw;
    }

    w = (*similarity)((real)T,(real)T,param);
    nw = (ht*(ht-1.0)*w)/2.0;
    assoce -= nw;
    associ += nw;

    for(i=T+1; i<=last; i++){
      hi = hist->Y[i];
      w = (*similarity)((real)i,(real)T,param);
      nw = hi*ht*w;
      assoce -= nw;
      cutie  += nw;
    }

    if(associ+cutie==0.0 || assoce+cutie==0.0)
      ncut = DBL_MAX;
    else
      ncut = cutie/(associ+cutie) + cutie/(assoce+cutie);
    
    F->Y[T-first] = ncut;
    F->X[T-first] = T;

    if(ncut<Copt){
      Copt = ncut;
      Topt = T;
    }
  }
  WriteCurve2Gnuplot(F,"curve_ncut.txt");
  DestroyCurve(&F);
  printf("Copt: %lf\n",Copt);
  return(Topt);
}


int    WatershedHistogramThreshold(Curve *hist){
  Curve *nhist;
  Set *S=NULL;
  int *cost,*label;
  int *data,*data1;
  double ymax;
  int i;

  nhist = NormalizeHistogram(hist);
  cost  = AllocIntArray(nhist->n);
  label = AllocIntArray(nhist->n);
  data  = AllocIntArray(nhist->n);
  data1 = AllocIntArray(nhist->n);

  /* Normalize Histogram within [0,4095] */
  ymax = CurveMaximum(nhist);
  for(i=0; i<nhist->n; i++){
    data[i] = (int)(4095*(nhist->Y[i]/ymax));
    data1[i] = data[i];
  }

  InsertSet(&S,0);
  InsertSet(&S,nhist->n-1);
  SupRec1(data,nhist->n,cost,label,S);
  DestroySet(&S);

  for(i=0; i<nhist->n; i++)
    data[i] = cost[i] - data[i];

  for(i=0; i<nhist->n-1; i++){
    if(label[i]!=label[i+1]){      
      InsertSet(&S,0);
      InsertSet(&S,nhist->n-1);
      break;
    }
  }

  SupRec1(data,nhist->n,cost,label,S);

  for(i=0; i<nhist->n-1; i++)
    if(label[i]!=label[i+1])
      break;

  DestroyCurve(&nhist);
  free(cost);
  free(label);
  free(data);
  free(data1);

  return(i); /* valley between peaks */
}


void SupRec1(int *data, int n, int *cost, int *label, Set *S){
  int i,p,q,cst;
  Set *seed;
  GQueue *Q=NULL;

  Q = CreateGQueue(4096,n,cost);

  for (i=0; i < n; i++) {
    cost[i]=INT_MAX;
    label[i]=0;
  }
    
  seed = S;
  while(seed != NULL){
    i = seed->elem;
    cost[i]=0;
    label[i]=i;
    InsertGQueue(&Q,i);
    seed = seed->next;
  }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
	
    q=p+1;
    if ((q < n-1)&&(label[q]==0)){
      cst = MAX(cost[p],data[q]);
      if (cst < cost[q]){
	cost[q]=cst;
	label[q]=label[p];
	InsertGQueue(&Q,q);
      }
    }

    q=p-1;
    if ((q >= 0)&&(label[q]==0)){
      cst = MAX(cost[p],data[q]);
      if (cst < cost[q]){
	cost[q]=cst;
	label[q]=label[p];
	InsertGQueue(&Q,q);
      }
    }
  }

  DestroyGQueue(&Q);
}




