
#include "methods.h"


Set *ForestRemoval(AdjRel3 *A, Scene *cost, 
		   Scene *pred, Scene *mark, 
		   Set **seedSet, Set **delSet){
  int i,p,q,mk,del_mk,n;
  Voxel u,v;
  BMap *Fcolor;
  Set **S, *tmp, *D;
  Set *frontier = NULL;
  Set *rootSet  = NULL;

  n = cost->xsize*cost->ysize*cost->zsize;
  Fcolor = BMapNew(n);
  BMapFill(Fcolor, 0);

  /* 
     Tenho que ler delSet e varrer seedSet removendo todas 
     sementes marcadas, setando custo para INT_MAX, etc.
     Tenho tambem que colocar essas raizes numa fila rootSet 
     para a etapa seguinte.
  */
  D = *delSet;
  while(D!=NULL){
    del_mk = D->elem;

    S = seedSet;
    while(*S!=NULL){
      p = (*S)->elem;
      mk = mark->data[p];
      // Sementes nao podem ser fronteira.(old)
      //_fast_BMapSet1(Fcolor, p); 

      if(mk == del_mk){
	InsertSet(&rootSet, p);
	cost->data[p] = INT_MAX;
	pred->data[p] = NIL;
	mark->data[p] = 0;
	tmp = *S;
	*S = (*S)->next;
	free(tmp);
      }
      else
	S = &((*S)->next);
    }
    D = D->next;
  }

  /*
     Varrer a partir de rootSet as arvores, resetando o 
     estado dos pixels e avaliando pixels de fronteira 
     que devem ser armazenados numa outra fila. Tenho que 
     ter um mapa binario para impedir que um pixel de 
     fronteira entre duas vezes ou mais na fila.
     Cuidado, pixels entre duas arvores de remocao nao 
     sao de fronteira!!!
  */
  while(rootSet!=NULL){
    p = RemoveSet(&rootSet);
    v.x = VoxelX(cost, p);
    v.y = VoxelY(cost, p);
    v.z = VoxelZ(cost, p);
    for(i=1; i<A->n; i++){
      u.x = v.x + A->dx[i];
      u.y = v.y + A->dy[i];
      u.z = v.z + A->dz[i];

      if(ValidVoxel(cost, u.x, u.y, u.z)){
	q = VoxelAddress(cost, u.x, u.y, u.z);
	mk = mark->data[q];	

	if(pred->data[q]==p){
	  InsertSet(&rootSet, q);
	  cost->data[q] = INT_MAX;
	  pred->data[q] = NIL;
	  // Para nao apagar label de sementes novas.
	  if( IsInSet(*delSet, mk) ) 
	    mark->data[q] = 0;
	}
	else if(_fast_BMapGet(Fcolor, q)==0){ 
	  if( !IsInSet(*delSet, mk) && cost->data[q] != INT_MAX ){ 
	    InsertSet(&frontier, q);
	    _fast_BMapSet1(Fcolor, q);
	  }
	}
      }
    }
  }
  BMapDestroy(Fcolor);
  DestroySet(delSet);

  return(frontier);
}


void RunDIFT(Scene *grad, Scene *cost, 
	     Scene *pred, Scene *mark, 
	     Set **seedSet, Set **delSet){
  PriorityQueue *Q=NULL;
  int i,p,q,n,xysize;
  Voxel u,v;
  int edge,tmp,Imax;
  AdjRel3 *A;
  Set *S, *F;

  xysize = grad->xsize*grad->ysize;
  n      = xysize*grad->zsize;
  Imax  = MaximumValue3(grad);
  Q     = CreatePQueue(Imax*2+1,n,cost->data);
  Q->C.minvalue = 0;
  A = Spheric(1.0);

  F = ForestRemoval(A, cost, pred, mark, seedSet, delSet);
  while(F != NULL){
    p = RemoveSet(&F);
    if(cost->data[p] != INT_MAX)
      FastInsertElemPQueue(Q,p);
  }

  S = *seedSet;
  while(S != NULL){
    p = S->elem;
    if( cost->data[p] != 0  || pred->data[p] != NIL ){
      if(Q->L.elem[p].color == GRAY)
	FastRemoveElemPQueue(Q,p);
      cost->data[p] = 0;
      pred->data[p] = NIL;
      FastInsertElemPQueue(Q,p);
    }
    S = S->next;
  }

  while(!IsEmptyPQueue(Q)){
    p = FastRemoveMinFIFOPQueue(Q);
    u.x = VoxelX(grad, p);
    u.y = VoxelY(grad, p);
    u.z = VoxelZ(grad, p);

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(grad,v.x,v.y,v.z)){
	q = VoxelAddress( grad, v.x, v.y, v.z );
	if(Q->L.elem[q].color != BLACK){
	  edge = grad->data[p] + grad->data[q];
	  tmp = MAX(cost->data[p],edge);
	  if((tmp < cost->data[q])||(pred->data[q] == p)){
	    if(Q->L.elem[q].color == GRAY)
	      FastRemoveElemPQueue(Q,q); // color of q = BLACK
	    cost->data[q] = tmp;
	    pred->data[q] = p;
	    mark->data[q] = mark->data[p];
	    //root->data[q] = root->data[p];
	    FastInsertElemPQueue(Q,q);
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  DestroyPQueue(&Q);
}



void RunIFT(Scene *grad, Scene *cost, 
	    Scene *pred, Scene *mark, 
	    Set *seedSet){
  PriorityQueue *Q=NULL;
  int i,p,q,n,xysize;
  Voxel u,v;
  int edge,tmp,Imax;
  AdjRel3 *A;
  Set *S;

  SetScene(pred, NIL);
  xysize = grad->xsize*grad->ysize;
  n      = xysize*grad->zsize;
  Imax  = MaximumValue3(grad);
  Q     = CreatePQueue(Imax*2+1,n,cost->data);
  Q->C.minvalue = 0;
  A = Spheric(1.0);

  S = seedSet;
  while(S != NULL){
    p = S->elem;
    cost->data[p] = 0;
    pred->data[p] = NIL;
    FastInsertElemPQueue(Q,p);
    S = S->next;
  }

  while(!IsEmptyPQueue(Q)){
    p = FastRemoveMinFIFOPQueue(Q);
    u.x = VoxelX(grad, p);
    u.y = VoxelY(grad, p);
    u.z = VoxelZ(grad, p);

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(grad,v.x,v.y,v.z)){
	q = VoxelAddress( grad, v.x, v.y, v.z );
	if(Q->L.elem[q].color != BLACK){
	  edge = grad->data[p] + grad->data[q];
	  tmp = MAX(cost->data[p],edge);
	  if(tmp < cost->data[q]){
	    if(Q->L.elem[q].color == GRAY)
	      FastRemoveElemPQueue(Q,q); // color of q = BLACK
	    cost->data[q] = tmp;
	    pred->data[q] = p;
	    mark->data[q] = mark->data[p];
	    FastInsertElemPQueue(Q,q);
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  DestroyPQueue(&Q);
}


//Returns the MST as a predecessor map (i.e., Scene *pred).
Scene *RunMST(Scene *grad, int root){
  PriorityQueue *Q=NULL;
  Scene *cost=NULL,*pred=NULL;
  int i,p,q,n;
  Voxel u,v;
  int edge,tmp,Imax;
  AdjRel3 *A;

  cost = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  pred = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  SetScene(cost, INT_MAX);
  SetScene(pred,     NIL);

  n    = grad->n;
  Imax = MaximumValue3(grad);
  Q    = CreatePQueue(Imax*2+1,n,cost->data);
  Q->C.minvalue = 0;
  A = Spheric(1.0);

  cost->data[root] = 0;
  pred->data[root] = NIL;
  FastInsertElemPQueue(Q,root);

  while(!IsEmptyPQueue(Q)){
    p = FastRemoveMinFIFOPQueue(Q);
    u.x = VoxelX(grad, p);
    u.y = VoxelY(grad, p);
    u.z = VoxelZ(grad, p);

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(grad,v.x,v.y,v.z)){
	q = VoxelAddress( grad, v.x, v.y, v.z );
	if(Q->L.elem[q].color != BLACK){
	  edge = grad->data[p] + grad->data[q];
	  tmp = edge;
	  if(tmp < cost->data[q]){
	    if(Q->L.elem[q].color == GRAY)
	      FastRemoveElemPQueue(Q,q); // color of q = BLACK
	    cost->data[q] = tmp;
	    pred->data[q] = p;
	    FastInsertElemPQueue(Q,q);
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  DestroyPQueue(&Q);
  DestroyScene(&cost);

  return pred;
}


//Returns the MST as a predecessor map (i.e., Scene *pred).
Scene *RunConstrainedMST(Scene *bin, Scene *grad){
  PriorityQueue *Q=NULL;
  Scene *cost=NULL,*pred=NULL;
  int i,r,p,q,n;
  Voxel u,v;
  int edge,tmp,Imax;
  AdjRel3 *A;

  cost = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  pred = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  SetScene(cost, INT_MAX);
  SetScene(pred,     NIL);

  n    = grad->n;
  Imax = MaximumValue3(grad);
  Q    = CreatePQueue(Imax*2+1,n,cost->data);
  Q->C.minvalue = 0;
  A = Spheric(1.0);

  for(r=0; r<n; r++){
    if(bin->data[r]==0) continue;
    else if(pred->data[r]!=NIL) continue;

    cost->data[r] = 0;
    pred->data[r] = NIL;
    FastInsertElemPQueue(Q,r);

    while(!IsEmptyPQueue(Q)){
      p = FastRemoveMinFIFOPQueue(Q);
      u.x = VoxelX(grad, p);
      u.y = VoxelY(grad, p);
      u.z = VoxelZ(grad, p);

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	v.z = u.z + A->dz[i];
	if (ValidVoxel(grad,v.x,v.y,v.z)){
	  q = VoxelAddress( grad, v.x, v.y, v.z );
	  if(bin->data[q]==0) continue;
	  if(Q->L.elem[q].color != BLACK){
	    edge = grad->data[p] + grad->data[q];
	    tmp = edge;
	    if(tmp < cost->data[q]){
	      if(Q->L.elem[q].color == GRAY)
		FastRemoveElemPQueue(Q,q); // color of q = BLACK
	      cost->data[q] = tmp;
	      pred->data[q] = p;
	      FastInsertElemPQueue(Q,q);
	    }
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  DestroyPQueue(&Q);
  DestroyScene(&cost);

  return pred;
}


void   ResumeFromScratchDIFT_Prototype(Scene *grad, Scene *label,
				       Scene *cost, Scene *pred, 
				       Scene *mark, Set **seedSet){
  Scene *mst=NULL;
  BMap *bmap=NULL,*isseed=NULL;
  int p,q,n,l1,l2,id=1;
  Set *S=NULL;


  SetScene(mark, 0);
  SetScene(pred, NIL);
  SetScene(cost, INT_MAX);

  n = grad->n;
  bmap   = BMapNew(n);
  isseed = BMapNew(n);
  mst  = RunMST(grad, 0);


  S = *seedSet;
  while(S != NULL){
    p = S->elem;
    _fast_BMapSet1(isseed,p);
    mark->data[p] = id;
    id++;
    S = S->next;
  }


  for(p=0; p<n; p++){
    q = p;
    while(mst->data[q]!=NIL &&
	  _fast_BMapGet(bmap,q)==0){
      l1 = label->data[q];
      l2 = label->data[mst->data[q]];
      if(l1!=l2){
	mark->data[q] = id;
	mark->data[mst->data[q]] = id+1;
	id+=2;
	if(_fast_BMapGet(isseed,q)==0){
	  InsertSet(seedSet, q);
	  _fast_BMapSet1(isseed,q);
	}
	if(_fast_BMapGet(isseed,mst->data[q])==0){
	  InsertSet(seedSet, mst->data[q]);
	  _fast_BMapSet1(isseed,mst->data[q]);
	}
      }
      _fast_BMapSet1(bmap,q);
      q = mst->data[q];
    }
    _fast_BMapSet1(bmap,q);
  }
  DestroyScene(&mst);
  BMapDestroy(bmap);
  BMapDestroy(isseed);

  ResumeIFT(label, grad,
	    cost, pred, 
	    mark, *seedSet);
}



void   ResumeFromScratchDIFT_MinSeeds(Scene *grad, Scene *label,
				      Scene *cost, Scene *pred, 
				      Scene *mark, Set **seedSet){
  Scene *energy,*bin,*bkg,*cores,*tmp;
  Set *Se = NULL;
  AdjRel3 *A;
  int l,Lmax,p,n;
  int Cmax,Mmax=0;


  SetScene(mark, 0);
  tmp    = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  energy = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  n = grad->n;
  A = Spheric(1.0);
  Lmax = MaximumValue3(label);


  for(l=0; l<=Lmax; l++){
    bin = Threshold3(label, l, l);
    bkg = Complement3(bin);
    Se  = Mask2Marker3(bkg, A);

    SetScene(pred, NIL);
    for(p=0; p<n; p++){
      if(bin->data[p]>0) energy->data[p] = INT_MAX;
      else               energy->data[p] = 0;
    }


    RunIFT(grad, energy,
	   pred, tmp, Se);

    cores = GetIFTCores(bin, grad, energy, A);

    //printf("GetCores: %d\n",MaximumValue3(cores));

    RefineIFTCores(cores, grad, energy, A);

    //printf("RefCores: %d\n",MaximumValue3(cores));

    RemoveRedundantIFTCores(cores, grad, energy, *seedSet, A);

    //printf("RemCores: %d\n",MaximumValue3(cores));

    Cmax = MaximumValue3(cores);
    for(p=0; p<n; p++){
      if(cores->data[p]>0)
	mark->data[p] = cores->data[p] + Mmax;
    }
    Mmax += Cmax;

    SelectSeedsInIFTCores(cores,seedSet);

    DestroyScene(&cores);
    DestroyScene(&bin);
    DestroyScene(&bkg);
    DestroySet(&Se);
  }
  DestroyScene(&energy);
  DestroyScene(&tmp);
  DestroyAdjRel3(&A);

  SetScene(pred, NIL);
  SetScene(cost, INT_MAX);

  ResumeIFT(label, grad,
	    cost, pred, 
	    mark, *seedSet);
}


Scene *GetIFTCores(Scene *bin, Scene *grad, 
		   Scene *energy, AdjRel3 *A){
  Scene *cores;
  int i,s,p,q,n,coreid=0;
  int edge,Es;
  Voxel u,v;
  FIFOQ *Q; 

  n = grad->n;
  Q = FIFOQNew(n);
  cores = CreateScene(grad->xsize, grad->ysize, grad->zsize);

  for(s=0; s<n; s++){
    if(bin->data[s]==0) continue;
    if(cores->data[s]>0) continue;

    coreid++;
    cores->data[s] = coreid;
    FIFOQPush(Q, s);
    Es = energy->data[s];

    while(!FIFOQEmpty(Q)){
      p = FIFOQPop(Q);
    
      u.x = VoxelX(grad, p);
      u.y = VoxelY(grad, p);
      u.z = VoxelZ(grad, p);
      for(i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	v.z = u.z + A->dz[i];
	if(ValidVoxel(grad,v.x,v.y,v.z)){
	  q = VoxelAddress(grad, v.x, v.y, v.z);
	  edge = grad->data[p] + grad->data[q];
	  if(edge<Es && cores->data[q]==0){
	    cores->data[q] = coreid;	    
	    FIFOQPush(Q, q);
	  }
	}
      }
    }
  }
  FIFOQDestroy(Q);
  return cores;
}



void RefineIFTCores(Scene *cores, Scene *grad, 
		    Scene *energy, AdjRel3 *A){
  int Cmax,c,cm,cM,i,p,q,n;
  int edge,Ep,Eq,cp,cq,coreid=0;
  int *equivalent,*hist,*tmp;
  Voxel u,v;

  n = grad->n;
  Cmax = MaximumValue3(cores);
  equivalent = AllocIntArray(Cmax+1);

  for(c=1; c<=Cmax; c++)
    equivalent[c] = c;

  for(p=0; p<n; p++){
    Ep = energy->data[p];
    cp = cores->data[p];
    if(cp==0) continue;
    u.x = VoxelX(grad, p);
    u.y = VoxelY(grad, p);
    u.z = VoxelZ(grad, p);
    for(i=1; i<A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if(ValidVoxel(grad,v.x,v.y,v.z)){
	q = VoxelAddress(grad, v.x, v.y, v.z);
	Eq = energy->data[q];
	cq = cores->data[q];
	edge = grad->data[p] + grad->data[q];

	if(equivalent[cp]!=equivalent[cq] && cq>0){
	  if(edge<=Ep && edge<=Eq){
	    cm = MIN(equivalent[cp],equivalent[cq]);
	    cM = MAX(equivalent[cp],equivalent[cq]);
	    for(c=1; c<=Cmax; c++)
	      if(equivalent[c]==cM)
		equivalent[c] = cm;
	  }
	}
      }
    }
  }
  hist = AllocIntArray(Cmax+1);
  for(c=1; c<=Cmax; c++)
    hist[equivalent[c]]++;

  coreid=0;
  tmp = AllocIntArray(Cmax+1);
  for(c=1; c<=Cmax; c++){
    if(hist[c]>0){
      coreid++;
      tmp[c] = coreid;
    }
  }
  for(c=1; c<=Cmax; c++)
    equivalent[c] = tmp[equivalent[c]];

  for(p=0; p<n; p++){
    cores->data[p] = equivalent[cores->data[p]];
  }
  free(tmp);
  free(hist);
  free(equivalent);  
}


void RemoveRedundantIFTCores(Scene *cores, Scene *grad, 
			     Scene *energy, Set *seedSet,
			     AdjRel3 *A){
  int Cmax,c,i,p,q,n;
  int edge,cp,cq,Ep,coreid=0;
  int *redundant,*tmp;
  int *marked=NULL;
  Set *S=NULL;
  Voxel u,v;

  n = grad->n;
  Cmax = MaximumValue3(cores);
  redundant = AllocIntArray(Cmax+1);
  marked    = AllocIntArray(Cmax+1);

  for(c=0; c<=Cmax; c++){
    redundant[c] = false;
    marked[c] = false;
  }
  S = seedSet;
  while(S != NULL){
    p = S->elem;
    marked[cores->data[p]] = true;
    S = S->next;
  }

  for(p=0; p<n; p++){
    Ep = energy->data[p];
    cp = cores->data[p];
    if(cp==0 || redundant[cp]==true) continue;
    u.x = VoxelX(grad, p);
    u.y = VoxelY(grad, p);
    u.z = VoxelZ(grad, p);
    for(i=1; i<A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if(ValidVoxel(grad,v.x,v.y,v.z)){
	q = VoxelAddress(grad, v.x, v.y, v.z);
	cq = cores->data[q];
	edge = grad->data[p] + grad->data[q];
	if(cp!=cq && cq>0){
	  if(edge<=Ep){
	    redundant[cp] = true;
	    break;
	  }
	}
      }
    }
  }
  coreid=0;
  tmp = AllocIntArray(Cmax+1);
  for(c=1; c<=Cmax; c++){
    if(redundant[c]==false || marked[c]==true){
      coreid++;
      tmp[c] = coreid;
    }
    else
      tmp[c] = 0;
  }
  for(p=0; p<n; p++){
    cores->data[p] = tmp[cores->data[p]];
  }
  free(tmp);
  free(marked);
  free(redundant);
}


//Take a seed for each core that has not yet been selected.
void SelectSeedsInIFTCores(Scene *cores, Set **seedSet){
  int Cmax,c,n,p;
  int *marked=NULL;
  Set *S=NULL;
  Voxel *C=NULL; //Central voxel of each core.
  int   *N=NULL; //Number of voxels in each core.
  int   *D=NULL; //Best distance to central node.
  int   *P=NULL; //Closest voxel to the center. 
  int dx,dy,dz,d;
  
  n = cores->n;
  Cmax = MaximumValue3(cores);
  marked = AllocIntArray(Cmax+1);
  C = (Voxel *) calloc(Cmax+1,sizeof(Voxel));
  N = AllocIntArray(Cmax+1);
  D = AllocIntArray(Cmax+1);
  P = AllocIntArray(Cmax+1);

  for(c=0; c<=Cmax; c++){
    marked[c] = false;
    C[c].x = 0;
    C[c].y = 0;
    C[c].z = 0;
    N[c] = 0;
    D[c] = INT_MAX;
    P[c] = NIL;
  }

  S = *seedSet;
  while(S != NULL){
    p = S->elem;
    marked[cores->data[p]] = true;
    S = S->next;
  }
  for(p=0; p<n; p++){
    c = cores->data[p];
    if(c==0 || marked[c]) continue;
    C[c].x += VoxelX(cores, p);
    C[c].y += VoxelY(cores, p);
    C[c].z += VoxelZ(cores, p);
    N[c]++;
  }
  for(c=1; c<=Cmax; c++){
    if(marked[c] || N[c]==0) continue;
    C[c].x = ROUND((float)C[c].x/(float)N[c]);
    C[c].y = ROUND((float)C[c].y/(float)N[c]);
    C[c].z = ROUND((float)C[c].z/(float)N[c]);
    p = VoxelAddress(cores,C[c].x,C[c].y,C[c].z);
    if(cores->data[p]==c){
      InsertSet(seedSet, p);
      marked[c] = true;
    }
  }
  for(p=0; p<n; p++){
    c = cores->data[p];
    if(c==0 || marked[c]) continue;
    dx = abs(C[c].x - VoxelX(cores, p));
    dy = abs(C[c].y - VoxelY(cores, p));
    dz = abs(C[c].z - VoxelZ(cores, p));
    d = MAX(MAX(dx,dy),dz);
    if(d<D[c]){
      D[c] = d;
      P[c] = p;
    }
  }
  for(c=1; c<=Cmax; c++){
    if(marked[c] || N[c]==0) continue;
    if(P[c]!=NIL){
      InsertSet(seedSet, P[c]);
      marked[c] = true;
    }
  }
  free(P);
  free(D);
  free(C);
  free(N);
  free(marked);
}


void   ResumeIFT(Scene *label, Scene *grad,
		 Scene *cost, Scene *pred, 
		 Scene *mark, Set *seedSet){
  PriorityQueue *Q=NULL;
  int i,p,q,n;
  Voxel u,v;
  int edge,tmp,Imax;
  AdjRel3 *A;
  Set *S;

  SetScene(pred, NIL);
  n    = grad->n;
  Imax = MaximumValue3(grad);
  Q    = CreatePQueue(Imax*2+1,n,cost->data);
  Q->C.minvalue = 0;
  A = Spheric(1.0);

  S = seedSet;
  while(S != NULL){
    p = S->elem;
    cost->data[p] = 0;
    pred->data[p] = NIL;
    FastInsertElemPQueue(Q,p);
    S = S->next;
  }

  while(!IsEmptyPQueue(Q)){
    p = FastRemoveMinFIFOPQueue(Q);
    u.x = VoxelX(grad, p);
    u.y = VoxelY(grad, p);
    u.z = VoxelZ(grad, p);

    for(i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(grad,v.x,v.y,v.z)){
	q = VoxelAddress( grad, v.x, v.y, v.z );
	if(Q->L.elem[q].color != BLACK &&
	   label->data[q]==label->data[p]){
	  edge = grad->data[p] + grad->data[q];
	  tmp = MAX(cost->data[p],edge);
	  if(tmp < cost->data[q]){
	    if(Q->L.elem[q].color == GRAY)
	      FastRemoveElemPQueue(Q,q); // color of q = BLACK
	    cost->data[q] = tmp;
	    pred->data[q] = p;
	    mark->data[q] = mark->data[p];
	    FastInsertElemPQueue(Q,q);
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  DestroyPQueue(&Q);
}






