
#include "rce.h"


/*"lf" -> The training set.*/
RCEClassifier *RCETraining(LabeledFeatures *lf,
			   real (*distance)(real*, real*, int)){
  RCEClassifier *rce;
  int Lmax,i,j,l,pos_2,p,q,nfeat;
  int *hist,*index,*pos;
  real r,rmin,*fv1,*fv2;

  Lmax = 0;
  for(i=0; i<lf->n; i++)
    if(lf->label[i]>Lmax)
      Lmax = lf->label[i];

  if(Lmax!=2){
    Warning("This version only works with two classes",
	    "RCETraining");
    return NULL;
  }

  hist = AllocIntArray(Lmax+1);
  pos  = AllocIntArray(Lmax+1);

  for(i=0; i<lf->n; i++)
    hist[lf->label[i]]++;

  pos[0] = 0;
  for(i=1; i<=Lmax; i++)
    pos[i] = pos[i-1]+hist[i-1];

  //Index to positions ordered by label.
  index = AllocIntArray(lf->n);
  for(i=0; i<lf->n; i++){
    l = lf->label[i];
    j = pos[l];
    index[j] = i;
    pos[l]++;
  }

  pos[0] = 0;
  for(i=1; i<=Lmax; i++)
    pos[i] = pos[i-1]+hist[i-1];

  rce = (RCEClassifier *)calloc(1, sizeof(RCEClassifier));
  if(rce == NULL)
    Error(MSG1,"RCETraining");
  
  nfeat = lf->fmap->nfeat;
  rce->n = hist[1];
  rce->fmap = CreateFeatMap(hist[1], nfeat);
  rce->radius = AllocRealArray(hist[1]);
  rce->distance = distance;

  pos_2 = pos[2];
  for(i=0; i<hist[1]; i++){
    p = index[pos[1]];
    fv1 = lf->fmap->data[p];
    memcpy(rce->fmap->data[i],
	   fv1,
	   nfeat*sizeof(real));
    rmin = REAL_MAX;
    pos[2] = pos_2;
    for(j=0; j<hist[2]; j++){
      q = index[pos[2]];
      fv2 = lf->fmap->data[q];
      r = (*distance)(fv1, fv2, nfeat);
      if(r<rmin)
	rmin = r;
      pos[2]++;
    }
    rce->radius[i] = rmin;
    pos[1]++;
  }

  free(index);
  free(hist);
  free(pos);

  return rce;
}


int   RCEClassifySample(RCEClassifier *rce,
			real *fv, int nfeat){
  int i;
  real *fv2,d;

  if(rce->fmap->nfeat!=nfeat)
    return 0;

  for(i=0; i<rce->n; i++){
    fv2 = rce->fmap->data[i];
    d = (*rce->distance)(fv, fv2, nfeat);
    if(d<rce->radius[i])
      return 1;
  }
  return 2;
}


void  RCEClassifySamples(RCEClassifier *rce,
			 LabeledFeatures *lf){
  int nfeat,i;
  real *fv;

  nfeat = lf->fmap->nfeat;
  for(i=0; i<lf->n; i++){
    fv = lf->fmap->data[i];
    lf->label[i] = RCEClassifySample(rce, fv, nfeat);
  }
}







