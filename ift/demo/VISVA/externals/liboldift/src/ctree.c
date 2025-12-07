#include "ctree.h"

int PassValue(CTree *ctree,int i,int j)
{
  while (i!=j){
    if (ctree->node[i].level <= ctree->node[j].level) 
      i = ctree->node[i].dad;
    else 
      j = ctree->node[j].dad;
  }
  return(ctree->node[i].level);
}

int Representative(Image *cmap, int p){
  if (cmap->val[p]==p)
    return(p);
  else
    return(cmap->val[p]=Representative(cmap,cmap->val[p]));
}


int Ancestor(Image *dad, Image *cmap, int rq)
{
  int r,ro;
    
  ro = r  = dad->val[rq];
  while (r != NIL) {
    ro = r = Representative(cmap,r);
    r  = dad->val[r];
  }
  return(ro);
}

CTree *CreateMaxTree(Image *img, AdjRel *A)
{
  CTree *ctree=(CTree *)calloc(1,sizeof(CTree));
  Image *dad,*cmap,*tmp,*level;
  int i,r,p,q,rp,rq,n,Imax=MaximumValue(img);
  GQueue *Q;
  Pixel u,v;
  int *nsons=NULL;
  int *size=NULL;

  ctree->cmap = CreateImage(img->ncols,img->nrows);
  cmap        = ctree->cmap;
  ctree->root = NIL; /* Tree is empty */
  dad         = CreateImage(img->ncols,img->nrows);
  level       = CreateImage(img->ncols,img->nrows);
  n           = img->ncols*img->nrows;
  size        = (int *)calloc(n,sizeof(int));    
  Q           = CreateGQueue(Imax+1,n,level->val);
  SetTieBreak(Q,LIFOBREAK);

  for (p=0; p < n; p++) {
    dad->val[p]  =NIL;
    cmap->val[p] =p;
    level->val[p]=Imax-img->val[p];
    size[p]=1;
    InsertGQueue(&Q,p);
  }
  
  while(!EmptyGQueue(Q)){
    p  = RemoveGQueue(Q);
    rp = Representative(cmap,p);
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (img->val[q]==img->val[p]){  /* propagate component */
	  if (Q->L.elem[q].color==GRAY){
	    cmap->val[q]=rp;
	    //if (p==rp) size[rp]=size[rp]+1; //Parece ser um bug
	    size[rp]=size[rp]+1;
	    UpdateGQueue(&Q,q,level->val[p]);
	  }	  
	} else {
	  if (img->val[p] < img->val[q]) /* find current dad of rq */
	    { 
	      rq = Representative(cmap,q);
	      r  = Ancestor(dad,cmap,rq);
	      if (r == NIL) { /* rp is dad of the rq */ 
		dad->val[rq]=rp;
	      } else {
		if (img->val[r]==img->val[rp]){ /* merge components */
		  if (r != rp) {		  
		    if (size[rp] <= size[r]){
		      cmap->val[rp] = r;
		      size[r] = size[r] + size[rp];
		      rp = r;
		    }else{
		      cmap->val[r] = rp;
		      size[rp] = size[rp] + size[r];		      
		    }
		  }
		} else { /* img->val[r] > img->val[rp] */		  
		  dad->val[r] = rp; /* rp is dad of r */ 
		}
	      }
	    }
	}
      }
    }
  }
  free(size);
  DestroyGQueue(&Q);
  DestroyImage(&level);

 /* Compress cmap map and count number of nodes */

  ctree->numnodes = 0; 
  for (p=0; p < n; p++) {
    if (dad->val[cmap->val[p]]!=NIL)
      r = cmap->val[p];
    cmap->val[p] = Representative(cmap,p);
    
    if (cmap->val[p]==p)
      ctree->numnodes++; 
  }

  /* Create and initialize nodes of the MaxTree. */

  ctree->node = (CTNode *)calloc(ctree->numnodes,sizeof(CTNode)); 
  tmp         = CreateImage(img->ncols,img->nrows);
  for (p=0; p < n; p++) {
    tmp->val[p]=NIL;
  }

  i = 0;
  for (p=0; p < n; p++) {
    if (cmap->val[p]==p){
      ctree->node[i].level = img->val[p];
      ctree->node[i].comp  = p;
      tmp->val[p]          = i;
      ctree->node[i].dad   = NIL;
      ctree->node[i].son   = NULL;
      ctree->node[i].numsons = 0;
      ctree->node[i].size  = 0;
      i++;
    }
  }

  /* Make the component map to point back to the maxtree. */

  for (p=0; p < n; p++) {
    if (tmp->val[p] == NIL)
      tmp->val[p] = tmp->val[cmap->val[p]];
  }

  for (p=0; p < n; p++) {
    cmap->val[p] = tmp->val[p];
  }
  DestroyImage(&tmp);

  /* Copy dad information to the maxtree and find its root */

  for (i=0; i < ctree->numnodes; i++) {
    if (dad->val[ctree->node[i].comp]!=NIL)
      ctree->node[i].dad = cmap->val[dad->val[ctree->node[i].comp]];
    else {
      ctree->node[i].dad = NIL;
      ctree->root = i;
    }
  }
 DestroyImage(&dad);

  /* Copy son information to the maxtree */
      
 nsons = (int *)calloc(ctree->numnodes,sizeof(int));
 for (i=0; i < ctree->numnodes; i++) {
   p = ctree->node[i].dad;
   if (p != NIL){
     nsons[p]++;
   }
 }
 for (i=0; i < ctree->numnodes; i++) {
   if (nsons[i] != 0){
     ctree->node[i].son = (int *)calloc(nsons[i],sizeof(int));
   }
 }
 free(nsons);

 for (i=0; i < ctree->numnodes; i++) {
   p = ctree->node[i].dad;
   if (p != NIL){
     ctree->node[p].son[ctree->node[p].numsons]=i;
     ctree->node[p].numsons++;
   }
 }

 /* Compute size of each node */

 for (p=0; p < n; p++) 
   ctree->node[cmap->val[p]].size++;
 
 return(ctree);
}

CTree *CreateMinTree(Image *img, AdjRel *A)
{
  CTree *ctree=(CTree *)calloc(1,sizeof(CTree));
  Image *dad,*cmap,*tmp;
  int i,r,p,q,rp,rq,n,Imax=MaximumValue(img);
  GQueue *Q;
  Pixel u,v;
  int *nsons=NULL;
  int *size=NULL;

  ctree->cmap = CreateImage(img->ncols,img->nrows);
  cmap        = ctree->cmap;
  ctree->root = NIL; /* Tree is empty */
  dad         = CreateImage(img->ncols,img->nrows);
  n           = img->ncols*img->nrows;
  size        = (int *)calloc(n,sizeof(int));    
  Q           = CreateGQueue(Imax+1,n,img->val);
  SetTieBreak(Q,LIFOBREAK);

  for (p=0; p < n; p++) {
    dad->val[p]  =NIL;
    cmap->val[p] =p;
    size[p]=1;
    InsertGQueue(&Q,p);
  }
  
  while(!EmptyGQueue(Q)){
    p  = RemoveGQueue(Q);
    rp = Representative(cmap,p);
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (img->val[q]==img->val[p]){  /* propagate component */
	  if (Q->L.elem[q].color==GRAY){
	    cmap->val[q]=rp;
	    //if (p==rp) size[rp]=size[rp]+1; Parece ser um bug
	    size[rp]=size[rp]+1;
	    UpdateGQueue(&Q,q,img->val[p]);
	  }	  
	} else {
	  if (img->val[p] > img->val[q]) /* find current dad of rq */
	    { 
	      rq = Representative(cmap,q);
	      r  = Ancestor(dad,cmap,rq);
	      if (r == NIL) { /* rp is dad of the rq */ 
		dad->val[rq]=rp;
	      } else {
		if (img->val[r]==img->val[rp]){ /* merge components */
		  if (r != rp) {		  
		    if (size[rp] <= size[r]){
		      cmap->val[rp] = r;
		      size[r] = size[r] + size[rp];
		      rp = r;
		    }else{
		      cmap->val[r] = rp;
		      size[rp] = size[rp] + size[r];		      
		    }
		  }
		} else { /* img->val[r] < img->val[rp] */		  
		  dad->val[r] = rp; /* rp is dad of r */ 
		}
	      }
	    }
	}
      }
    }
  }
  free(size);
  DestroyGQueue(&Q);

 /* Compress cmap map and count number of nodes */

  ctree->numnodes = 0; 
  for (p=0; p < n; p++) {
    if (dad->val[cmap->val[p]]!=NIL)
      r = cmap->val[p];
    cmap->val[p] = Representative(cmap,p);
    
    if (cmap->val[p]==p)
      ctree->numnodes++; 
  }

  /* Create and initialize nodes of the MinTree. */

  ctree->node = (CTNode *)calloc(ctree->numnodes,sizeof(CTNode)); 
  tmp         = CreateImage(img->ncols,img->nrows);
  for (p=0; p < n; p++) {
    tmp->val[p]=NIL;
  }

  i = 0;
  for (p=0; p < n; p++) {
    if (cmap->val[p]==p){
      ctree->node[i].level = img->val[p];
      ctree->node[i].comp  = p;
      tmp->val[p]          = i;
      ctree->node[i].dad   = NIL;
      ctree->node[i].son  = NULL;
      ctree->node[i].numsons = 0;
      ctree->node[i].size  = 0;
      i++;
    }
  }

  /* Make the component map to point back to the mintree. */

  for (p=0; p < n; p++) {
    if (tmp->val[p] == NIL)
      tmp->val[p] = tmp->val[cmap->val[p]];
  }

  for (p=0; p < n; p++) {
    cmap->val[p] = tmp->val[p];
  }
  DestroyImage(&tmp);

  /* Copy dad information to the mintree and find its root */

  for (i=0; i < ctree->numnodes; i++) {
    if (dad->val[ctree->node[i].comp]!=NIL)
      ctree->node[i].dad = cmap->val[dad->val[ctree->node[i].comp]];
    else {
      ctree->node[i].dad = NIL;
      ctree->root = i;
    }
  }
 DestroyImage(&dad);

  /* Copy son information to the mintree */
      
 nsons = (int *)calloc(ctree->numnodes,sizeof(int));
 for (i=0; i < ctree->numnodes; i++) {
   p = ctree->node[i].dad;
   if (p != NIL){
     nsons[p]++;
   }
 }
 for (i=0; i < ctree->numnodes; i++) {
   if (nsons[i] != 0){
     ctree->node[i].son = (int *)calloc(nsons[i],sizeof(int));
   }
 }
 free(nsons);

 for (i=0; i < ctree->numnodes; i++) {
   p = ctree->node[i].dad;
   if (p != NIL){
     ctree->node[p].son[ctree->node[p].numsons]=i;
     ctree->node[p].numsons++;
   }
 }

 /* Compute size of each node */

 for (p=0; p < n; p++) 
   ctree->node[cmap->val[p]].size++;
 
 return(ctree);
}

void DestroyCTree(CTree **ctree)
{
  CTree *tmp=*ctree;
  int i;

  if (tmp != NULL) {
    DestroyImage(&(tmp->cmap));
    for (i=0; i < tmp->numnodes; i++){
      if (tmp->node[i].numsons!=0)
	free(tmp->node[i].son);
    }
    free(tmp->node);
    free(tmp);
    *ctree = NULL;
  }  
}

AnnImg *TopoWater(Image *img, AdjRel *A, int flag)
{
  GQueue *Q=NULL;
  int i,p,q,n,pass_value,border;
  Pixel u,v;
  AnnImg *aimg=NULL;
  Image *cost,*root,*pred,*divide;
  CTree *ctree=NULL;

  aimg   = CreateAnnImg(img);
  cost   = aimg->cost;
  root   = aimg->label;
  pred   = aimg->pred;
  n      = img->ncols*img->nrows;
  Q      = CreateGQueue(QSIZE+1,n,cost->val);
  ctree  = CreateMinTree(img,A);

  /* Classical watershed */

  for (p=0; p < n; p++) {
    i = ctree->cmap->val[p];
    if (ctree->node[i].numsons == 0){ /* p belongs to a minimum */
      root->val[p]  = i; 
      pred->val[p]  = NIL; 
      cost->val[p]  = img->val[p];
      InsertGQueue(&Q,p);
    } else {
      pred->val[p]  = NIL; 
      root->val[p]  = NIL; 
      cost->val[p]  = INT_MAX;
    }
  }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (cost->val[q]==INT_MAX){
	  cost->val[q]=MAX(cost->val[p],img->val[q]);
	  root->val[q]=root->val[p];
	  pred->val[q]=p; 
	  InsertGQueue(&Q,q);
	}
      }
    }
  }
  DestroyGQueue(&Q);

  /* Assigment of pass values */

  divide = CreateImage(img->ncols,img->nrows);

  for (p=0; p < n; p++) {
    u.x = p%img->ncols;
    u.y = p/img->ncols; 
    cost->val[p]=INT_MIN;
    border=0;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (root->val[p]<root->val[q]){
	  border=1;
	  divide->val[p]=1;
	  pass_value = PassValue(ctree,root->val[p],root->val[q]);
	  if (pass_value > cost->val[p])
	    cost->val[p]=pass_value;	  
	}
      }
    }
    if ((!border)||(cost->val[p]==INT_MIN))
      cost->val[p]=0;//ctree->node[root->val[p]].level;
  }

  WriteImage(divide,"divide.pgm");
  DestroyImage(&divide);

  DestroyCTree(&ctree);

  return(aimg);
}

AnnImg *TopoWater_Laurent(Image *img, AdjRel *A, int flag)
{
  GQueue *Q=NULL;
  int i,p,q,n;
  Pixel u,v;
  AnnImg *aimg=NULL;
  Image *cost,*root,*pred;
  CTree *ctree=NULL;
  int pass_value, tmp_pv;
  bool *water_divide=NULL,low;
  Image *tmp;
  
  ctree = CreateMinTree(img,A);
  aimg  = CreateAnnImg(img);
  cost  = aimg->cost;
  root  = aimg->label;
  pred  = aimg->pred;
  n     = img->ncols*img->nrows;
  water_divide = (bool *)calloc(n,sizeof(bool));
  Q     = CreateGQueue(QSIZE+1,n,cost->val);

  for (p=0; p < n; p++) {
    i = ctree->cmap->val[p];
    water_divide[p]=false;
    if (ctree->node[i].numsons == 0){ /* p belongs to a minimum */
      pred->val[p]  = NIL;
      root->val[p]  = i; 
      cost->val[p]  = img->val[p];
      InsertGQueue(&Q,p);
    } else {
      pred->val[p]  = NIL;
      root->val[p] = NIL; 
      cost->val[p]  = INT_MAX;
    }
  }

  while(!EmptyGQueue(Q)) {
    p=RemoveGQueue(Q);
    pass_value = INT_MIN; 
    u.x = p%img->ncols;
    u.y = p/img->ncols;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q = v.x + img->tbrow[v.y];
	if (cost->val[q] != INT_MAX){ /* q has been visited */
	  if ((root->val[q] != root->val[p])&&
	      (Q->L.elem[q].color == BLACK)){ /* p is in WD */ 
	    water_divide[p] = true;
	    if (cost->val[q] < cost->val[p]){ /* select
						 maximum
						 pass
						 value */
	      tmp_pv = PassValue(ctree,root->val[p],root->val[q]);
	      if (tmp_pv > pass_value)
		pass_value = tmp_pv;
	    }
	  }
	}else{ /* q has not been visited yet */
	  cost->val[q]=MAX(cost->val[p],img->val[q]);
	  root->val[q]=root->val[p];
	  pred->val[q]=p;
	  InsertGQueue(&Q,q);
	}
      }
    }

    if (water_divide[p]){
      if (pass_value != INT_MIN){ /* set pass_value of p */
	cost->val[p]=pass_value;
      }else{ /* propagate pass_value of the predecessor of p */
	cost->val[p] = ctree->node[root->val[p]].level;
      }
    }else{ /* propagate pass_value of the predecessor of p */
      cost->val[p] = ctree->node[root->val[p]].level;
    }  
    
    if (flag==2)
      if (pred->val[p]!=NIL)
	if (cost->val[p] < cost->val[pred->val[p]])
	  cost->val[p]=cost->val[pred->val[p]];
  }
  
  
  tmp = CreateImage(img->ncols,img->nrows);

  for (p=0; p < n; p++) {
    if (water_divide[p]){      
      tmp->val[p]=1;
      low = true;  
      pass_value = INT_MAX;
      u.x = p%img->ncols;
      u.y = p/img->ncols;
      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(img,v.x,v.y)){
	  q = v.x + img->tbrow[v.y];
	  if (cost->val[q] < pass_value) 
	    pass_value = cost->val[q];
	  if (!water_divide[q]){
	    if (root->val[p]!=root->val[q]){
	      low = false;
	      break;
	    }
	  }else{
	    if (cost->val[p] > cost->val[q]){
	      low = false;
	      break;
	    }
	  }
	}
      }
      if (low){
	  cost->val[p] = pass_value;
	  water_divide[p]=0;
	  tmp->val[p]=0;
      }
    }
  }
  
  WriteImage(tmp,"divide.pgm");
  DestroyImage(&tmp);
  
  free(water_divide);
  DestroyGQueue(&Q);
  DestroyCTree(&ctree);

  return(aimg);
}

void CumSize(CTree *ctree, int i)
{
  int s,j;

  for (j=0; j < ctree->node[i].numsons; j++){
    s = ctree->node[i].son[j];
    CumSize(ctree,s);
    ctree->node[i].size = ctree->node[i].size + ctree->node[s].size;
  } 	      
  return;
}

int AreaLevel(CTree *ctree, int *level, int i, int thres)
{
  if ((ctree->node[i].size > thres)||(i==ctree->root))
    return(ctree->node[i].level);
  else
    return(level[i]=AreaLevel(ctree,level,ctree->node[i].dad,thres));
}

Image *CTAreaClose(Image *img, int thres)
{
  CTree *ctree=NULL;
  int i,p,n;
  Image *fimg;
  AdjRel *A=Circular(1.0);
  int *level=NULL;

  ctree = CreateMinTree(img,A);
  CumSize(ctree,ctree->root);
  level = AllocIntArray(ctree->numnodes);
  for (i=0; i < ctree->numnodes; i++) 
    level[i]=ctree->node[i].level;

  for (i=0; i < ctree->numnodes; i++) 
    if (ctree->node[i].numsons==0)
      level[i]=AreaLevel(ctree,level,i,thres);
  fimg = CreateImage(img->ncols,img->nrows);
  n    = img->ncols*img->nrows;
  for (p=0; p < n; p++) 
    fimg->val[p]=level[ctree->cmap->val[p]];
  DestroyCTree(&ctree);
  free(level);
  DestroyAdjRel(&A);
  return(fimg);
}

Image *CTAreaOpen(Image *img, int thres)
{
  CTree *ctree=NULL;
  int i,p,n;
  Image *fimg;
  AdjRel *A=Circular(1.0);
  int *level=NULL;

  ctree = CreateMaxTree(img,A);
  CumSize(ctree,ctree->root);
  level = AllocIntArray(ctree->numnodes);
  for (i=0; i < ctree->numnodes; i++) 
    level[i]=ctree->node[i].level;
  
  for (i=0; i < ctree->numnodes; i++) 
    if (ctree->node[i].numsons==0)
      level[i]=AreaLevel(ctree,level,i,thres);
  fimg = CreateImage(img->ncols,img->nrows);
  n    = img->ncols*img->nrows;
  for (p=0; p < n; p++) 
    fimg->val[p]=level[ctree->cmap->val[p]];
  DestroyCTree(&ctree);
  free(level);
  DestroyAdjRel(&A);
  return(fimg);
}

int VolumeLevel(CTree *ctree, int *level, int i, int thres, int cumvol)
{
  int dad,vol=cumvol;

  dad = ctree->node[i].dad;
  if (dad != NIL)
    vol = cumvol+
      abs(ctree->node[i].level-ctree->node[dad].level)*ctree->node[i].size;

  if ((vol > thres)||(i==ctree->root))
    return(ctree->node[i].level);
  else
    return(level[i]=VolumeLevel(ctree,level,dad,thres,vol));
}

Image *CTVolumeClose(Image *img, int thres)
{
  CTree *ctree=NULL;
  int i,p,n;
  Image *fimg;
  AdjRel *A=Circular(1.0);
  int *level=NULL;

  ctree = CreateMinTree(img,A);
  CumSize(ctree,ctree->root);
  level = AllocIntArray(ctree->numnodes);
  for (i=0; i < ctree->numnodes; i++) 
    level[i]=ctree->node[i].level;
  for (i=0; i < ctree->numnodes; i++) 
    if (ctree->node[i].numsons==0)
      level[i]=VolumeLevel(ctree,level,i,thres,0);    
  fimg = CreateImage(img->ncols,img->nrows);
  n    = img->ncols*img->nrows;
  for (p=0; p < n; p++) 
    fimg->val[p]=level[ctree->cmap->val[p]];
  DestroyCTree(&ctree);
  free(level);
  DestroyAdjRel(&A);
  return(fimg);
}

Image *CTVolumeOpen(Image *img, int thres)
{
  CTree *ctree=NULL;
  int i,p,n;
  Image *fimg;
  AdjRel *A=Circular(1.0);
  int *level=NULL;

  ctree = CreateMaxTree(img,A);
  CumSize(ctree,ctree->root);
  level = AllocIntArray(ctree->numnodes);
  for (i=0; i < ctree->numnodes; i++) 
    level[i]=ctree->node[i].level;
  for (i=0; i < ctree->numnodes; i++) 
    if (ctree->node[i].numsons==0)
      level[i]=VolumeLevel(ctree,level,i,thres,0);
  fimg = CreateImage(img->ncols,img->nrows);
  n    = img->ncols*img->nrows;
  for (p=0; p < n; p++) 
    fimg->val[p]=level[ctree->cmap->val[p]];
  DestroyCTree(&ctree);
  free(level);
  DestroyAdjRel(&A);
  return(fimg);
}
