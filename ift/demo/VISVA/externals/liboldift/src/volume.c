#include "volume.h"


Volume  *CreateVolume(int ncols,int nrows,int nimgs)
{
  Volume *vol=NULL;
  int i,n;

  vol = (Volume *) calloc(1,sizeof(Volume));
  if (vol == NULL){
    Error(MSG1,"CreateVolume");
  }

  vol->val   = AllocIntArray(nrows*ncols*nimgs);
  vol->tbrow = AllocIntArray(nrows);
  vol->tbimg = AllocIntArray(ncols*nrows);
  
  vol->tbrow[0]=0;
  for (i=1; i < nrows; i++)
    vol->tbrow[i]=vol->tbrow[i-1] + ncols;

  vol->tbimg[0]=0; n = ncols*nrows;
  for (i=1; i < nimgs; i++)
    vol->tbimg[i]=vol->tbimg[i-1] + n;

  vol->ncols = ncols;
  vol->nrows = nrows;
  vol->nimgs = nimgs;

  return(vol);
}

void     DestroyVolume(Volume **vol)
{
  Volume *aux;

  aux = *vol;
  if(aux != NULL){
    if (aux->val != NULL)   free(aux->val); 
    if (aux->tbrow != NULL) free(aux->tbrow);
    if (aux->tbimg != NULL) free(aux->tbimg);
    free(aux);    
    *vol = NULL;
  }
}

Volume *Lift(Image *img)
{
  Volume *vol=NULL;
  int x,y,z,i;

  vol = CreateVolume(img->ncols,img->nrows,MaximumValue(img));
  for (y=0; y < img->nrows; y++)
    for (x=0; x < img->ncols; x++){
      i = x+img->tbrow[y];
      for (z=1; z < img->val[i]; z++)
      	vol->val[i+vol->tbimg[z]]=1;
    }
  return(vol);
}

Image *Flat(Volume *vol)
{
  Image *img=NULL;
  int x,y,z,i;

  img = CreateImage(vol->ncols,vol->nrows);
  for (z=0; z < vol->nimgs; z++)
    for (y=0; y < vol->nrows; y++)
      for (x=0; x < vol->ncols; x++){
	i = x+vol->tbrow[y];
	if (vol->val[i+vol->tbimg[z]]==1)
	  img->val[i] = z+1;
      }
  return(img);
}

Volume *CopyVolume(Volume *vol)
{
  Volume *volc;

  volc = CreateVolume(vol->ncols,vol->nrows,vol->nimgs);
  memcpy(volc->val,vol->val,vol->ncols*vol->nrows*vol->nimgs*sizeof(int));
  
  return(volc);
}

void SetVolume(Volume *vol, int value)
{
  int p,n;

  n = vol->nrows*vol->ncols*vol->nimgs;
  for (p=0; p < n; p++)
    vol->val[p]=value;  
}

int VolumeMax(Volume *vol)
{
  int i,n,m;
  m = vol->val[0];
  n = VolumeLen(vol);
  for(i=1;i<n;i++)
    if (vol->val[i] > m)
      m = vol->val[i];
  return m;
}


