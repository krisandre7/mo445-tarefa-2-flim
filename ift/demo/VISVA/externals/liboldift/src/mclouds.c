#include "mclouds.h"

MClouds *CreateMClouds(int n)
{
  MClouds *mclouds=NULL;
  int i;

  mclouds = (MClouds *) calloc(1,sizeof(MClouds));
  if (mclouds != NULL){
    mclouds->A = (AdjRel **) calloc(n,sizeof(AdjRel *)); 
    for (i=0; i < n; i++) 
      mclouds->A[i]=NULL;
  }
  mclouds->n=n;

  return(mclouds);
}

void DestroyMClouds(MClouds **mclouds)
{
  MClouds *aux;
  int i;

  aux = *mclouds;
  if (aux != NULL){
    for (i=0; i < aux->n; i++) 
      if (aux->A[i]!=NULL)
	DestroyAdjRel(&aux->A[i]);
    free(aux);
    *mclouds = NULL;
  }
}

MClouds *RoundEye(float r1, float r2, float r3)
{
  MClouds *mclouds=CreateMClouds(2);

  mclouds->A[0]=Circular(r1);
  mclouds->A[1]=Ring(r2,r3);

  return(mclouds);
}

MClouds *TwoBalls(float r1, float r2, int dx, int dy)
{
  MClouds *mclouds=CreateMClouds(2);
  int i;

  mclouds->A[0]=Circular(r1);
  mclouds->A[1]=Circular(r2);
  mclouds->n=2;
  for (i=0; i < mclouds->A[1]->n; i++){
    mclouds->A[1]->dx[i] += dx;
    mclouds->A[1]->dy[i] += dy;
  }

  return(mclouds);
}

Image *DrawMClouds(MClouds *mclouds)
{
  Image *img=NULL;
  int    x,y,ncols,nrows,i,j,dx_min,dx_max,dy_min,dy_max,q;

  dx_min=INT_MAX;
  dx_max=INT_MIN;
  dy_min=INT_MAX;
  dy_max=INT_MIN;

  for (i=0; i < mclouds->n; i++) {
    for (j=0; j < mclouds->A[i]->n; j++){ 
      if (mclouds->A[i]->dx[j] < dx_min)
	dx_min = mclouds->A[i]->dx[j];
      if (mclouds->A[i]->dx[j] > dx_max)
	dx_max = mclouds->A[i]->dx[j];
      if (mclouds->A[i]->dy[j] < dy_min)
	dy_min = mclouds->A[i]->dy[j];
      if (mclouds->A[i]->dy[j] > dy_max)
	dy_max = mclouds->A[i]->dy[j];
    }
  }
  
  /* create image with zeros around the clouds */

  ncols = 2*(dx_max - dx_min + 1); 
  nrows = 2*(dy_max - dy_min + 1);
  
  img = CreateImage(ncols,nrows);

  for (i=0; i < mclouds->n; i++) {
    for (j=0; j < mclouds->A[i]->n; j++){ 
      x = ncols/2 + mclouds->A[i]->dx[j];
      y = nrows/2 + mclouds->A[i]->dy[j];
      q = x + img->tbrow[y];
      img->val[q] = i+1;
    }
  }
  
  return(img);
}
