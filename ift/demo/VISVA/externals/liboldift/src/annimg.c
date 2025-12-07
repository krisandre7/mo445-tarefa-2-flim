#include "annimg.h"
#include "common.h"

AnnImg *CreateAnnImg(Image *img)
{
  AnnImg *aimg=NULL;

  aimg = (AnnImg *) calloc(1,sizeof(AnnImg));
  if (aimg == NULL)
    Error(MSG1,"CreateAnnImg");

  aimg->img   = img;
  aimg->cost  = CreateImage(img->ncols,img->nrows);    
  aimg->label = CreateImage(img->ncols,img->nrows);
  aimg->pred  = CreateImage(img->ncols,img->nrows);
  aimg->root  = CreateImage(img->ncols,img->nrows);
  aimg->grad  = CreateImage(img->ncols,img->nrows);

  return(aimg);
}

void  DestroyAnnImg(AnnImg **aimg)
{
  AnnImg *aux;
  
  aux = *aimg;
  if (aux != NULL){
    DestroyImage(&(aux->cost));
    DestroyImage(&(aux->label));
    DestroyImage(&(aux->pred));
    DestroyImage(&(aux->root));
    DestroyImage(&(aux->grad));
    free(aux);
    *aimg = NULL;
  }
}

AnnImg *Annotate(Image *img, Image *cost, Image *label)
{
  AnnImg *aimg=NULL;
  int p,n;

  aimg = (AnnImg *) calloc(1,sizeof(AnnImg));
  if (aimg == NULL)
    Error(MSG1,"Annotate");

  aimg->img   = img;
  aimg->cost  = CreateImage(img->ncols,img->nrows);    
  aimg->label = CreateImage(img->ncols,img->nrows);
  aimg->pred  = CreateImage(img->ncols,img->nrows);
  aimg->root = NULL;
  aimg->seed  = NULL;

  n = img->ncols*img->nrows;

  if ((cost == NULL)&&(label == NULL))
    for (p=0; p < n; p++){
      aimg->cost->val[p]  = INT_MAX;
      aimg->label->val[p] = 0;
      aimg->pred->val[p]  = p;
    }
  else
    if ((cost == NULL)&&(label != NULL))
      for (p=0; p < n; p++){
	aimg->pred->val[p]  = p;
	if (label->val[p] > 0) {
	  aimg->cost->val[p]  = 0;
	  aimg->label->val[p] = label->val[p];
	  InsertSet(&(aimg->seed),p);
	} else {
	  aimg->cost->val[p]  = INT_MAX;
	  aimg->label->val[p] = 0;
	}
      }
    else
      if ((cost != NULL)&&(label == NULL))
	for (p=0; p < n; p++){
	  aimg->cost->val[p]  = cost->val[p];
	  aimg->label->val[p] = p;
	  aimg->pred->val[p]  = p;
	  InsertSet(&(aimg->seed),p);
	}
      else
	if ((cost != NULL)&&(label != NULL))	
	  for (p=0; p < n; p++){
	    aimg->pred->val[p]  = p;
	    aimg->label->val[p] = label->val[p];
	    if (label->val[p] > 0) {
	      aimg->cost->val[p]  = cost->val[p];
	      InsertSet(&(aimg->seed),p);
	    } else { 
	      aimg->cost->val[p]  = INT_MAX;
	    }
	  }
  
  return(aimg);
}

void   DeAnnotate(AnnImg **aimg)
{
  AnnImg *aux;
  
  aux = *aimg;
  if (aux != NULL){
    DestroyImage(&(aux->cost));
    DestroyImage(&(aux->label));
    DestroyImage(&(aux->pred));
    DestroySet(&(aux->seed));
    free(aux);
    *aimg = NULL;
  }
}

void AddSeed(AnnImg *aimg, int pixel, int cost, int label, int pred)
{
  InsertSet(&(aimg->seed),pixel);
  aimg->cost->val[pixel]   = cost;
  aimg->label->val[pixel]  = label;
  aimg->pred->val[pixel]   = pred;
}

bool RemSeed(AnnImg *aimg, AdjRel *A, int pixel) /* adj. relation must
                                                    be the same used
                                                    to propagate the
                                                    seed pixel */
{
  int i,p,q,n;
  Pixel u,v;
  Image *pred,*color;
  
  if (aimg->pred->val[pixel] != pixel) /* pixel must be a seed */
    return(false);

  color = CreateImage(aimg->img->ncols,aimg->img->nrows);
  pred  = CompPaths(aimg->pred);
  n = aimg->img->ncols*aimg->img->nrows;
  for (p=0; p < n; p++) {
    if (pred->val[p]==pixel){
      aimg->cost->val[p] = INT_MAX;
      aimg->label->val[p]= 0;
      aimg->pred->val[p] = p;
      u.x = p%aimg->img->ncols;
      u.y = p/aimg->img->ncols;
      for (i=0; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(aimg->img,v.x,v.y)){
	  q = v.x + aimg->img->tbrow[v.y];
	  if ((pred->val[q] != pixel)&&(color->val[q]==WHITE)){
	    InsertSet(&(aimg->seed),q);
	    color->val[q]=BLACK;
	  }
	}
      }
    }
  }

  DestroyImage(&pred);
  DestroyImage(&color);
  return(true);
}

Image *CompPaths(Image *pred)
{
  Image *seed=NULL;
  int p,n;

  seed = ift_CopyImage(pred);

  n = seed->ncols*seed->nrows;
  for (p=0; p < n; p++) 
    seed->val[p] = Seed(seed,p);
  return(seed);
}

int Seed(Image *pred, int p)
{
  if (pred->val[p]==p)
    return(p);
  else 
    return(Seed(pred,pred->val[p]));    
}

int SeedComp(Image *pred, int p)
{

  if (pred->val[p]==p)
    return(p);
  else {
    pred->val[p] = SeedComp(pred,pred->val[p]);
    return(pred->val[p]);
  }

}

int *Path(Image *pred, int dst)
{
  int *path=NULL,i,n,p;
  
  p = dst;
  n = 1;
  while(pred->val[p]!=p){
    n++;
    p = pred->val[p];
  }
  
  path = AllocIntArray(n+1);
  path[0]=n;
  p = dst;
  i = 0;
  while(pred->val[p]!=p){
    i++;
    path[i]=p;
    p = pred->val[p];
  }
  i++;
  path[i]=p;

  return(path);
}

Image *GetCost(AnnImg *aimg)
{
  return(ift_CopyImage(aimg->cost));
}

Image *GetLabel(AnnImg *aimg)
{
  return(ift_CopyImage(aimg->label));
}

Image *GetPred(AnnImg *aimg)
{
  return(ift_CopyImage(aimg->pred));
}

Image *Label2Root(Image *label)
{
  int p,n;
  Image *root=NULL;
  
  root = CreateImage(label->ncols,label->nrows);
  n    = label->ncols*label->nrows;
  for (p=0; p < n; p++) 
    if (label->val[p] > 0)
      root->val[p] = p;
    else
      root->val[p] = NIL;
  return(root);
}

Image *Root2Label(Image *root, Image *rootlabel)
{
  int p,n;
  Image *label=NULL;
  
  label = CreateImage(root->ncols,root->nrows);
  n    = label->ncols*label->nrows;
  for (p=0; p < n; p++) 
    if (root->val[p] != NIL)
      label->val[p] = rootlabel->val[root->val[p]];
    else
      label->val[p] = 0;
  return(label);
}
