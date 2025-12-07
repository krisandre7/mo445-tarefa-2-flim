#include "ctree3.h"

int Representative3(Scene *cmap, int p){
  if (cmap->data[p]==p)
    return(p);
  else
    return(cmap->data[p]=Representative3(cmap,cmap->data[p]));
}

int Ancestor3(Scene *dad, Scene *cmap, int rq)
{
  int r,ro;
    
  ro = r  = dad->data[rq];
  while (r != NIL) {
    ro = r = Representative3(cmap,r);
    r  = dad->data[r];
  }
  return(ro);
}

CTree3 *CreateMaxTree3(Scene *scn, AdjRel3 *A)
{
  CTree3 *ctree=(CTree3 *)calloc(1,sizeof(CTree3));
  Scene  *dad,*cmap,*tmp,*level;
  int     i,r,p,q,rp,rq,n,Imax=MaximumValue3(scn);
  GQueue *Q;
  Voxel   u,v;
  int *nsons=NULL,sz=scn->xsize*scn->ysize;
  int *size=NULL;

  ctree->cmap = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  cmap        = ctree->cmap;
  ctree->root = NIL; /* Tree is empty */
  dad         = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  level       = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n           = sz*scn->zsize;
  size        = (int *)calloc(n,sizeof(int));    
  Q           = CreateGQueue(Imax+1,n,level->data);
  SetTieBreak(Q,LIFOBREAK);

  for (p=0; p < n; p++) {
    dad->data[p]  =NIL;
    cmap->data[p] =p;
    level->data[p]=Imax-scn->data[p];
    size[p]=1;
    InsertGQueue(&Q,p);
  }
  
  while(!EmptyGQueue(Q)){
    p  = RemoveGQueue(Q);
    rp = Representative3(cmap,p);
    u.z =  p/sz;
    u.x = (p%sz)%scn->xsize;
    u.y = (p%sz)/scn->xsize;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (scn->data[q]==scn->data[p]){  /* propagate component */
	  if (Q->L.elem[q].color==GRAY){
	    cmap->data[q]=rp;
	    if (p==rp) size[rp]=size[rp]+1;
	    UpdateGQueue(&Q,q,level->data[p]);
	  }	  
	} else {
	  if (scn->data[p] < scn->data[q]) /* find current dad of rq */
	    { 
	      rq = Representative3(cmap,q);
	      r  = Ancestor3(dad,cmap,rq);
	      if (r == NIL) { /* rp is dad of the rq */ 
		dad->data[rq]=rp;
	      } else {
		if (scn->data[r]==scn->data[rp]){ /* merge components */
		  if (r != rp) {		  
		    if (size[rp] <= size[r]){
		      cmap->data[rp] = r;
		      size[r] = size[r] + size[rp];
		      rp = r;
		    }else{
		      cmap->data[r] = rp;
		      size[rp] = size[rp] + size[r];		      
		    }
		  }
		} else { /* scn->data[r] > scn->data[rp] */		  
		  dad->data[r] = rp; /* rp is dad of r */ 
		}
	      }
	    }
	}
      }
    }
  }
  free(size);
  DestroyGQueue(&Q);
  DestroyScene(&level);

 /* Compress cmap map and count number of nodes */

  ctree->numnodes = 0; 
  for (p=0; p < n; p++) {
    if (dad->data[cmap->data[p]]!=NIL)
      r = cmap->data[p];
    cmap->data[p] = Representative3(cmap,p);
    
    if (cmap->data[p]==p)
      ctree->numnodes++; 
  }

  /* Create and initialize nodes of the MaxTree. */

  ctree->node = (CTNode *)calloc(ctree->numnodes,sizeof(CTNode)); 
  tmp         = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (p=0; p < n; p++) {
    tmp->data[p]=NIL;
  }

  i = 0;
  for (p=0; p < n; p++) {
    if (cmap->data[p]==p){
      ctree->node[i].level = scn->data[p];
      ctree->node[i].comp  = p;
      tmp->data[p]         = i;
      ctree->node[i].dad   = NIL;
      ctree->node[i].son   = NULL;
      ctree->node[i].numsons = 0;
      ctree->node[i].size  = 0;
      i++;
    }
  }

  /* Make the component map to point back to the maxtree. */

  for (p=0; p < n; p++) {
    if (tmp->data[p] == NIL)
      tmp->data[p] = tmp->data[cmap->data[p]];
  }

  for (p=0; p < n; p++) {
    cmap->data[p] = tmp->data[p];
  }
  DestroyScene(&tmp);

  /* Copy dad information to the maxtree and find its root */

  for (i=0; i < ctree->numnodes; i++) {
    if (dad->data[ctree->node[i].comp]!=NIL)
      ctree->node[i].dad = cmap->data[dad->data[ctree->node[i].comp]];
    else {
      ctree->node[i].dad = NIL;
      ctree->root = i;
    }
  }
 DestroyScene(&dad);

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
   ctree->node[cmap->data[p]].size++;
 
 return(ctree);
}

CTree3 *CreateMinTree3(Scene *scn, AdjRel3 *A)
{
  CTree3 *ctree=(CTree3 *)calloc(1,sizeof(CTree3));
  Scene  *dad,*cmap,*tmp;
  int i,r,p,q,rp,rq,n,Imax=MaximumValue3(scn);
  GQueue *Q;
  Voxel u,v;
  int *nsons=NULL,sz=scn->xsize*scn->ysize;
  int *size=NULL;

  ctree->cmap = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  cmap        = ctree->cmap;
  ctree->root = NIL; /* Tree is empty */
  dad         = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n           = sz*scn->zsize;
  size        = (int *)calloc(n,sizeof(int));    
  Q           = CreateGQueue(Imax+1,n,scn->data);
  SetTieBreak(Q,LIFOBREAK);

  for (p=0; p < n; p++) {
    dad->data[p]  =NIL;
    cmap->data[p] =p;
    size[p]=1;
    InsertGQueue(&Q,p);
  }
  printf("cheguei\n");
  
  while(!EmptyGQueue(Q)){
    p  = RemoveGQueue(Q);
    rp = Representative3(cmap,p);
    u.z =  p/sz;
    u.x = (p%sz)%scn->xsize;
    u.y = (p%sz)/scn->xsize;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (scn->data[q]==scn->data[p]){  /* propagate component */
	  if (Q->L.elem[q].color==GRAY){
	    cmap->data[q]=rp;
	    if (p==rp) size[rp]=size[rp]+1;
	    UpdateGQueue(&Q,q,scn->data[p]);
	  }	  
	} else {
	  if (scn->data[p] > scn->data[q]) /* find current dad of rq */
	    { 
	      rq = Representative3(cmap,q);
	      r  = Ancestor3(dad,cmap,rq);
	      if (r == NIL) { /* rp is dad of the rq */ 
		dad->data[rq]=rp;
	      } else {
		if (scn->data[r]==scn->data[rp]){ /* merge components */
		  if (r != rp) {		  
		    if (size[rp] <= size[r]){
		      cmap->data[rp] = r;
		      size[r] = size[r] + size[rp];
		      rp = r;
		    }else{
		      cmap->data[r] = rp;
		      size[rp] = size[rp] + size[r];		      
		    }
		  }
		} else { /* scn->data[r] < scn->data[rp] */		  
		  dad->data[r] = rp; /* rp is dad of r */ 
		}
	      }
	    }
	}
      }
    }
  }
  free(size);
  DestroyGQueue(&Q);

  printf("isso demora?\n");
  
 /* Compress cmap map and count number of nodes */

  ctree->numnodes = 0; 
  for (p=0; p < n; p++) {
    if (dad->data[cmap->data[p]]!=NIL)
      r = cmap->data[p];
    cmap->data[p] = Representative3(cmap,p);
    
    if (cmap->data[p]==p)
      ctree->numnodes++; 
  }

  /* Create and initialize nodes of the MaxTree. */

  ctree->node = (CTNode *)calloc(ctree->numnodes,sizeof(CTNode)); 
  tmp         = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (p=0; p < n; p++) {
    tmp->data[p]=NIL;
  }

  i = 0;
  for (p=0; p < n; p++) {
    if (cmap->data[p]==p){
      ctree->node[i].level = scn->data[p];
      ctree->node[i].comp  = p;
      tmp->data[p]         = i;
      ctree->node[i].dad   = NIL;
      ctree->node[i].son   = NULL;
      ctree->node[i].numsons = 0;
      ctree->node[i].size  = 0;
      i++;
    }
  }

  /* Make the component map to point back to the maxtree. */

  for (p=0; p < n; p++) {
    if (tmp->data[p] == NIL)
      tmp->data[p] = tmp->data[cmap->data[p]];
  }

  for (p=0; p < n; p++) {
    cmap->data[p] = tmp->data[p];
  }
  DestroyScene(&tmp);

  /* Copy dad information to the maxtree and find its root */

  for (i=0; i < ctree->numnodes; i++) {
    if (dad->data[ctree->node[i].comp]!=NIL)
      ctree->node[i].dad = cmap->data[dad->data[ctree->node[i].comp]];
    else {
      ctree->node[i].dad = NIL;
      ctree->root = i;
    }
  }
 DestroyScene(&dad);

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
   ctree->node[cmap->data[p]].size++;
 
 return(ctree);
}

void DestroyCTree3(CTree3 **ctree)
{
  CTree3 *tmp=*ctree;
  int i;

  if (tmp != NULL) {
    DestroyScene(&(tmp->cmap));
    for (i=0; i < tmp->numnodes; i++){
      if (tmp->node[i].numsons!=0)
	free(tmp->node[i].son);
    }
    free(tmp->node);
    free(tmp);
    *ctree = NULL;
  }  
}

void CumSize3(CTree3 *ctree, int i)
{
  int s,j;

  for (j=0; j < ctree->node[i].numsons; j++){
    s = ctree->node[i].son[j];
    CumSize3(ctree,s);
    ctree->node[i].size = ctree->node[i].size + ctree->node[s].size;
  } 	      
  return;
}

int AreaLevel3(CTree3 *ctree, int *level, int i, int thres)
{
  if ((ctree->node[i].size > thres)||(i==ctree->root))
    return(ctree->node[i].level);
  else
    return(level[i]=AreaLevel3(ctree,level,ctree->node[i].dad,thres));
}

Scene *CTAreaClose3(Scene *scn, int thres)
{
  CTree3 *ctree=NULL;
  int i,p,n;
  Scene *fscn;
  AdjRel3 *A=Spheric(1.0);
  int *level=NULL;

  ctree = CreateMinTree3(scn,A);
  CumSize3(ctree,ctree->root);
  level = AllocIntArray(ctree->numnodes);
  for (i=0; i < ctree->numnodes; i++) 
    level[i]=ctree->node[i].level;

  for (i=0; i < ctree->numnodes; i++) 
    if (ctree->node[i].numsons==0)
      level[i]=AreaLevel3(ctree,level,i,thres);
  fscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n    = scn->xsize*scn->ysize*scn->zsize;
  for (p=0; p < n; p++) 
    fscn->data[p]=level[ctree->cmap->data[p]];
  DestroyCTree3(&ctree);
  free(level);
  DestroyAdjRel3(&A);
  return(fscn);
}

Scene *CTAreaOpen3(Scene *scn, int thres)
{
  CTree3 *ctree=NULL;
  int i,p,n;
  Scene *fscn;
  AdjRel3 *A=Spheric(1.0);
  int *level=NULL;

  ctree = CreateMaxTree3(scn,A);
  CumSize3(ctree,ctree->root);
  level = AllocIntArray(ctree->numnodes);
  for (i=0; i < ctree->numnodes; i++) 
    level[i]=ctree->node[i].level;
  
  for (i=0; i < ctree->numnodes; i++) 
    if (ctree->node[i].numsons==0)
      level[i]=AreaLevel3(ctree,level,i,thres);
  fscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n    = scn->xsize*scn->ysize*scn->zsize;
  for (p=0; p < n; p++) 
    fscn->data[p]=level[ctree->cmap->data[p]];
  DestroyCTree3(&ctree);
  free(level);
  DestroyAdjRel3(&A);
  return(fscn);
}

int VolumeLevel3(CTree3 *ctree, int *level, int i, int thres, int cumvol)
{
  int dad,vol=cumvol;

  dad = ctree->node[i].dad;
  if (dad != NIL)
    vol = cumvol+
      abs(ctree->node[i].level-ctree->node[dad].level)*ctree->node[i].size;

  if ((vol > thres)||(i==ctree->root))
    return(ctree->node[i].level);
  else
    return(level[i]=VolumeLevel3(ctree,level,dad,thres,vol));
}

Scene *CTVolumeClose3(Scene *scn, int thres)
{
  CTree3 *ctree=NULL;
  int i,p,n;
  Scene   *fscn;
  AdjRel3 *A=Spheric(1.0);
  int     *level=NULL;

  ctree = CreateMinTree3(scn,A);
  CumSize3(ctree,ctree->root);
  level = AllocIntArray(ctree->numnodes);
  for (i=0; i < ctree->numnodes; i++) 
    level[i]=ctree->node[i].level;
  for (i=0; i < ctree->numnodes; i++) 
    if (ctree->node[i].numsons==0)
      level[i]=VolumeLevel3(ctree,level,i,thres,0);    
  fscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n    = scn->xsize*scn->ysize*scn->zsize;
  for (p=0; p < n; p++) 
    fscn->data[p]=level[ctree->cmap->data[p]];
  DestroyCTree3(&ctree);
  free(level);
  DestroyAdjRel3(&A);
  return(fscn);
}

Scene *CTVolumeOpen3(Scene *scn, int thres)
{
  CTree3 *ctree=NULL;
  int i,p,n;
  Scene *fscn;
  AdjRel3 *A=Spheric(1.0);
  int *level=NULL;

  ctree = CreateMaxTree3(scn,A);
  CumSize3(ctree,ctree->root);
  level = AllocIntArray(ctree->numnodes);
  for (i=0; i < ctree->numnodes; i++) 
    level[i]=ctree->node[i].level;
  for (i=0; i < ctree->numnodes; i++) 
    if (ctree->node[i].numsons==0)
      level[i]=VolumeLevel3(ctree,level,i,thres,0);
  fscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n    = scn->xsize*scn->ysize*scn->zsize;
  for (p=0; p < n; p++) 
    fscn->data[p]=level[ctree->cmap->data[p]];
  DestroyCTree3(&ctree);
  free(level);
  DestroyAdjRel3(&A);
  return(fscn);
}
