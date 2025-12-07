
#include "labeledfeatures.h"

/*All data assigned with this method will be 
  automatically deleted by "DestroyLabeledFeatures" as 
  appropriate (i.e. it takes ownership of the fmap).*/
LabeledFeatures *CreateLabeledFeatures(FeatMap *fmap){
  LabeledFeatures *lf;

  lf = (LabeledFeatures *)calloc(1, sizeof(LabeledFeatures));
  if(lf == NULL)
    Error(MSG1,"CreateLabeledFeatures");
  lf->n = fmap->n;
  lf->label = AllocIntArray(fmap->n);
  lf->fmap = fmap;
  return lf;
}

void DestroyLabeledFeatures(LabeledFeatures **lf){
  LabeledFeatures *aux;

  aux = *lf;
  if(aux != NULL){
    if(aux->label != NULL)
      free(aux->label);
    if(aux->fmap != NULL)
      DestroyFeatMap(&(aux->fmap));
    free(aux);
    *lf = NULL;
  }
}


/*Creates a copy of the original dataset.*/
LabeledFeatures *CloneLabeledFeatures(LabeledFeatures *lf){
  LabeledFeatures *clone;
  FeatMap *fmap;

  if(lf==NULL) return NULL;
  fmap = CloneFeatMap(lf->fmap);
  clone = CreateLabeledFeatures(fmap);
  memcpy(clone->label,
	 lf->label,
	 lf->n*sizeof(int));
  return clone;
}

/*Write the dataset to disk in binary format.*/
void  WriteLabeledFeatures(LabeledFeatures *lf, 
			   char *filename){
  char msg[512];
  int n,nfeat,i;
  unsigned char size;
  FILE *fp;
  
  fp = fopen(filename,"wb");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"WriteLabeledFeatures");
  }

  size     = (unsigned char)sizeof(real);
  n        = lf->n;
  nfeat    = lf->fmap->nfeat;
  fwrite(&n,        sizeof(int), 1, fp);
  fwrite(&nfeat,    sizeof(int), 1, fp);
  fwrite(&size,     sizeof(unsigned char), 1, fp);
  fwrite(lf->label, sizeof(int), n, fp);
  for(i=0; i<n; i++)
    fwrite(lf->fmap->data[i], sizeof(real), nfeat, fp);

  fclose(fp);
}

/*Read a dataset from the disk in binary format.*/
LabeledFeatures *ReadLabeledFeatures(char *filename){
  LabeledFeatures *lf;
  FeatMap *fmap;
  char msg[512];
  int n,nfeat,i,j,p;
  unsigned char size;
  float  *faux=NULL;
  double *daux=NULL;
  FILE *fp;
  real *fv;

  fp = fopen(filename,"rb");
  if(fp == NULL){
    sprintf(msg,"Cannot open %s",filename);
    Error(msg,"ReadLabeledFeatures");
  }

  fread(&n,        sizeof(int), 1, fp);
  fread(&nfeat,    sizeof(int), 1, fp);
  fread(&size,     sizeof(unsigned char), 1, fp);
  fmap = CreateFeatMap(n, nfeat);
  lf   = CreateLabeledFeatures(fmap);
  fread(lf->label, sizeof(int), n, fp);

  if(size==sizeof(real)){
    for(i=0; i<n; i++)
      fread(lf->fmap->data[i], sizeof(real), nfeat, fp);
  }
  else if(size==sizeof(float)){
    faux = AllocFloatArray(n);
    fread(faux, sizeof(float), n*nfeat, fp);
    p = 0;
    for(i=0; i<n; i++){
      fv = lf->fmap->data[i];
      for(j=0; j<nfeat; j++){
	fv[j] = (real)faux[p];
	p++;
      }
    }
    free(faux);
  }
  else if(size==sizeof(double)){
    daux = AllocDoubleArray(n);
    fread(daux, sizeof(double), n*nfeat, fp);
    p = 0;
    for(i=0; i<n; i++){
      fv = lf->fmap->data[i];
      for(j=0; j<nfeat; j++){
	fv[j] = (real)daux[p];
	p++;
      }
    }
    free(daux);
  }
  else
    Error("Bad or corrupted file",
	  "ReadLabeledFeatures");

  fclose(fp);

  return lf;
}

/*It shuffles the samples at random.*/
void RandomizeLabeledFeatures(LabeledFeatures *lf){
  int i,j,tmp;
  real *fv_tmp;
  
  srand((int)time(NULL));
  for(i=0; i<lf->n; i++){
    j = RandomInteger(0, lf->n-1);

    tmp = lf->label[i];
    lf->label[i] = lf->label[j];
    lf->label[j] = tmp;

    fv_tmp = lf->fmap->data[i];
    lf->fmap->data[i] = lf->fmap->data[j];
    lf->fmap->data[j] = fv_tmp;
  }
}

/*Returns a subset with samples of all classes in the 
  proportion given by "rate". This subset is removed from the
  original dataset. You should call "RandomizeLabeledFeatures"
  first to ensure the selection of random samples.*/
LabeledFeatures *RemoveLabeledFeaturesSamples(LabeledFeatures **lf, 
					      float rate){
  LabeledFeatures *lf0 = NULL,*lf1 = NULL,*lf2 = NULL;
  FeatMap *fmap;
  int *hist,*hist1,*hist2,*index,*pos;
  int Lmax,i,j,l,nsamples,nfeat,p1,p2;
  
  lf0 = *lf;
  Lmax = 0;
  for(i=0; i<lf0->n; i++)
    if(lf0->label[i]>Lmax)
      Lmax = lf0->label[i];
  hist  = AllocIntArray(Lmax+1);
  hist1 = AllocIntArray(Lmax+1);
  hist2 = AllocIntArray(Lmax+1);
  pos   = AllocIntArray(Lmax+1);

  for(i=0; i<lf0->n; i++)
    hist[lf0->label[i]]++;

  nsamples = 0;
  for(i=0; i<=Lmax; i++){
    hist1[i] = ceil(hist[i]*rate);
    hist2[i] = hist[i] - hist1[i];
    nsamples += hist1[i];
  }

  pos[0] = 0;
  for(i=1; i<=Lmax; i++)
    pos[i] = pos[i-1]+hist[i-1];

  nfeat = lf0->fmap->nfeat;
  fmap = CreateFeatMap(nsamples, nfeat);
  lf1 = CreateLabeledFeatures(fmap);
  fmap = CreateFeatMap(lf0->n-nsamples, nfeat);
  lf2 = CreateLabeledFeatures(fmap);

  //Index to positions ordered by label.
  index = AllocIntArray(lf0->n);
  for(i=0; i<lf0->n; i++){
    l = lf0->label[i];
    j = pos[l];
    index[j] = i;
    pos[l]++;
  }

  p1 = p2 = 0;
  for(l=0; l<=Lmax; l++){
    for(i=0; i<hist1[l]; i++){
      memcpy(lf1->fmap->data[p1],
	     lf0->fmap->data[index[p1+p2]],
	     nfeat*sizeof(real));
      lf1->label[p1] = lf0->label[index[p1+p2]];
      p1++;
    }
    for(j=0; j<hist2[l]; j++){
      memcpy(lf2->fmap->data[p2],
	     lf0->fmap->data[index[p1+p2]],
	     nfeat*sizeof(real));
      lf2->label[p2] = lf0->label[index[p1+p2]];
      p2++;
    }
  }

  //Changes lf e lf2.
  *lf = lf2;
  DestroyLabeledFeatures(&lf0);

  free(index);
  free(hist);
  free(hist1);
  free(hist2);
  free(pos);
  
  return lf1;
}


/*Concatenate two datasets.*/
LabeledFeatures *MergeLabeledFeatures(LabeledFeatures *lf1, 
				      LabeledFeatures *lf2){
  LabeledFeatures *lf;
  FeatMap *fmap;
  int n,nfeat,i;
  
  if(lf1==NULL && lf2==NULL) return NULL;
  if(lf1==NULL)
    return CloneLabeledFeatures(lf2);
  if(lf2==NULL)
    return CloneLabeledFeatures(lf1);

  n = lf1->n + lf2->n;
  nfeat = lf1->fmap->nfeat;
  if(nfeat!=lf2->fmap->nfeat) return NULL;

  fmap = CreateFeatMap(n, nfeat);
  lf = CreateLabeledFeatures(fmap);

  memcpy(lf->label,
	 lf1->label,
	 lf1->n*sizeof(int));
  memcpy(&(lf->label[lf1->n]),
	 lf2->label,
	 lf2->n*sizeof(int));

  for(i=0; i<lf1->n; i++)
    memcpy(lf->fmap->data[i],
	   lf1->fmap->data[i],
	   nfeat*sizeof(real));

  for(i=0; i<lf2->n; i++)
    memcpy(lf->fmap->data[i+lf1->n],
	   lf2->fmap->data[i],
	   nfeat*sizeof(real));
  
  return lf;
}

