
#include "bia_ift.h"


Set *ForestRemoval(bia::AdjRel3::AdjRel3 *A, 
		   bia::Scene16::Scene16 *cost, 
		   Scene *pred, Scene *mark, 
		   Set **seedSet, Set **delSet){
  int i,p,q,mk,del_mk,n;
  bia::Voxel u,v;
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
	cost->data[p] = USHRT_MAX;
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
    v.c.x = bia::Scene16::GetAddressX(cost, p);
    v.c.y = bia::Scene16::GetAddressY(cost, p);
    v.c.z = bia::Scene16::GetAddressZ(cost, p);
    for(i=1; i<A->n; i++){
      u.v = v.v + A->d[i].v;

      if(bia::Scene16::IsValidVoxel(cost, u)){
	q = bia::Scene16::GetVoxelAddress(cost, u);
	mk = mark->data[q];	

	if(pred->data[q]==p){
	  InsertSet(&rootSet, q);
	  cost->data[q] = USHRT_MAX;
	  pred->data[q] = NIL;
	  // Para nao apagar label de sementes novas.
	  if( IsInSet(*delSet, mk) ) 
	    mark->data[q] = 0;
	}
	else if(_fast_BMapGet(Fcolor, q)==0){ 
	  if( !IsInSet(*delSet, mk) && cost->data[q] != USHRT_MAX ){ 
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


void RunDIFT(bia::Scene16::Scene16 *grad, 
	     bia::Scene16::Scene16 *cost, 
	     Scene *pred, Scene *mark, 
	     Set **seedSet, Set **delSet){
  bia::PQueue16::PQueue16 *Q=NULL;
  int i,p,q,n,xysize;
  bia::Voxel u,v;
  ushort edge,tmp,Imax;
  bia::AdjRel3::AdjRel3 *A;
  Set *S, *F;

  xysize = grad->xsize*grad->ysize;
  n      = xysize*grad->zsize;
  Imax  = bia::Scene16::GetMaximumValue(grad);
  Q     = bia::PQueue16::Create(Imax+1, //Imax*2+1,
				n,cost->data);
  Q->C.minvalue = 0;
  A = bia::AdjRel3::Spheric(1.0);

  F = ForestRemoval(A, cost, pred, mark, seedSet, delSet);
  while(F != NULL){
    p = RemoveSet(&F);
    if(cost->data[p] != USHRT_MAX)
      bia::PQueue16::FastInsertElem(Q,p);
  }

  S = *seedSet;
  while(S != NULL){
    p = S->elem;
    if( cost->data[p] != 0  || pred->data[p] != NIL ){
      if(Q->L.elem[p].color == GRAY)
	bia::PQueue16::FastRemoveElem(Q,p);
      cost->data[p] = 0;
      pred->data[p] = NIL;
      bia::PQueue16::FastInsertElem(Q,p);
    }
    S = S->next;
  }

  while(!bia::PQueue16::IsEmpty(Q)){
    p = bia::PQueue16::FastRemoveMinFIFO(Q);
    u.c.x = bia::Scene16::GetAddressX(grad, p);
    u.c.y = bia::Scene16::GetAddressY(grad, p);
    u.c.z = bia::Scene16::GetAddressZ(grad, p);

    for(i=1; i<A->n; i++){
      v.v = u.v + A->d[i].v;
      if(bia::Scene16::IsValidVoxel(grad,v)){
	q = bia::Scene16::GetVoxelAddress(grad, v);

	if(Q->L.elem[q].color != BLACK){
	  edge = (ushort)((((int)grad->data[p]) + 
			   ((int)grad->data[q]))>>1);

	  tmp = MAX(cost->data[p],edge);
	  if((tmp < cost->data[q])||(pred->data[q] == p)){
	    if(Q->L.elem[q].color == GRAY) 
	      bia::PQueue16::FastRemoveElem(Q,q);
	    // color of q = BLACK
	    cost->data[q] = tmp;
	    pred->data[q] = p;
	    mark->data[q] = mark->data[p];
	    //root->data[q] = root->data[p];
	    bia::PQueue16::FastInsertElem(Q,q);
	  }
	}
      }
    }
  }
  bia::AdjRel3::Destroy(&A);
  bia::PQueue16::Destroy(&Q);
}



void RunIFT(bia::Scene16::Scene16 *grad, 
	    bia::Scene16::Scene16 *cost, 
	    Scene *pred, Scene *mark, 
	    Set *seedSet){
  bia::PQueue16::PQueue16 *Q=NULL;
  int i,p,q,n,xysize;
  bia::Voxel u,v;
  ushort edge,tmp,Imax;
  bia::AdjRel3::AdjRel3 *A;
  Set *S;

  SetScene(pred, NIL);
  xysize = grad->xsize*grad->ysize;
  n      = xysize*grad->zsize;
  Imax  = bia::Scene16::GetMaximumValue(grad);
  Q     = bia::PQueue16::Create(Imax+1, //Imax*2+1,
				n,cost->data);
  Q->C.minvalue = 0;
  A = bia::AdjRel3::Spheric(1.0);

  S = seedSet;
  while(S != NULL){
    p = S->elem;
    cost->data[p] = 0;
    pred->data[p] = NIL;
    bia::PQueue16::FastInsertElem(Q,p);
    S = S->next;
  }

  while(!bia::PQueue16::IsEmpty(Q)){
    p = bia::PQueue16::FastRemoveMinFIFO(Q);
    u.c.x = bia::Scene16::GetAddressX(grad, p);
    u.c.y = bia::Scene16::GetAddressY(grad, p);
    u.c.z = bia::Scene16::GetAddressZ(grad, p);

    for(i=1; i<A->n; i++){
      v.v = u.v + A->d[i].v;
      if(bia::Scene16::IsValidVoxel(grad,v)){
	q = bia::Scene16::GetVoxelAddress(grad, v);

	if(Q->L.elem[q].color != BLACK){
	  edge = (ushort)((((int)grad->data[p]) + 
			   ((int)grad->data[q]))>>1);

	  tmp = MAX(cost->data[p],edge);
	  if(tmp < cost->data[q]){
	    if(Q->L.elem[q].color == GRAY)
	      bia::PQueue16::FastRemoveElem(Q,q);
	    // color of q = BLACK
	    cost->data[q] = tmp;
	    pred->data[q] = p;
	    mark->data[q] = mark->data[p];
	    bia::PQueue16::FastInsertElem(Q,q);
	  }
	}
      }
    }
  }
  bia::AdjRel3::Destroy(&A);
  bia::PQueue16::Destroy(&Q);
}


//Returns the MST as a predecessor map (i.e., Scene *pred).
Scene *RunMST(bia::Scene16::Scene16 *grad, int root){
  bia::PQueue16::PQueue16 *Q=NULL;
  bia::Scene16::Scene16 *cost=NULL;
  Scene *pred=NULL;
  int i,p,q,n;
  bia::Voxel u,v;
  ushort edge,tmp,Imax;
  bia::AdjRel3::AdjRel3 *A;

  cost = bia::Scene16::Create(grad->xsize, 
			      grad->ysize, 
			      grad->zsize);
  pred = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  bia::Scene16::Fill(cost, USHRT_MAX);
  SetScene(pred,     NIL);

  n    = grad->n;
  Imax = bia::Scene16::GetMaximumValue(grad);
  Q    = bia::PQueue16::Create(Imax+1, //Imax*2+1,
			       n,cost->data);
  Q->C.minvalue = 0;
  A = bia::AdjRel3::Spheric(1.0);

  cost->data[root] = 0;
  pred->data[root] = NIL;
  bia::PQueue16::FastInsertElem(Q,root);

  while(!bia::PQueue16::IsEmpty(Q)){
    p = bia::PQueue16::FastRemoveMinFIFO(Q);
    u.c.x = bia::Scene16::GetAddressX(grad, p);
    u.c.y = bia::Scene16::GetAddressY(grad, p);
    u.c.z = bia::Scene16::GetAddressZ(grad, p);

    for(i=1; i<A->n; i++){
      v.v = u.v + A->d[i].v;
      if(bia::Scene16::IsValidVoxel(grad,v)){
	q = bia::Scene16::GetVoxelAddress(grad, v);

	if(Q->L.elem[q].color != BLACK){
	  edge = (ushort)((((int)grad->data[p]) + 
			   ((int)grad->data[q]))>>1);
	  tmp = edge;
	  if(tmp < cost->data[q]){
	    if(Q->L.elem[q].color == GRAY)
	      bia::PQueue16::FastRemoveElem(Q,q);
	    // color of q = BLACK
	    cost->data[q] = tmp;
	    pred->data[q] = p;
	    bia::PQueue16::FastInsertElem(Q,q);
	  }
	}
      }
    }
  }
  bia::AdjRel3::Destroy(&A);
  bia::PQueue16::Destroy(&Q);
  bia::Scene16::Destroy(&cost);

  return pred;
}


//Returns the MST as a predecessor map (i.e., Scene *pred).
Scene *RunConstrainedMST(Scene *bin, bia::Scene16::Scene16 *grad){
  bia::PQueue16::PQueue16 *Q=NULL;
  bia::Scene16::Scene16 *cost=NULL;
  Scene *pred=NULL;
  int i,r,p,q,n;
  bia::Voxel u,v;
  ushort edge,tmp,Imax;
  bia::AdjRel3::AdjRel3 *A;

  cost = bia::Scene16::Create(grad->xsize, 
			      grad->ysize, 
			      grad->zsize);
  pred = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  bia::Scene16::Fill(cost, USHRT_MAX);
  SetScene(pred,     NIL);

  n    = grad->n;
  Imax = bia::Scene16::GetMaximumValue(grad);
  Q    = bia::PQueue16::Create(Imax+1, //Imax*2+1,
			       n,cost->data);
  Q->C.minvalue = 0;
  A = bia::AdjRel3::Spheric(1.0);

  for(r=0; r<n; r++){
    if(bin->data[r]==0) continue;
    else if(pred->data[r]!=NIL) continue;

    cost->data[r] = 0;
    pred->data[r] = NIL;
    bia::PQueue16::FastInsertElem(Q,r);

    while(!bia::PQueue16::IsEmpty(Q)){
      p = bia::PQueue16::FastRemoveMinFIFO(Q);
      u.c.x = bia::Scene16::GetAddressX(grad, p);
      u.c.y = bia::Scene16::GetAddressY(grad, p);
      u.c.z = bia::Scene16::GetAddressZ(grad, p);

      for(i=1; i<A->n; i++){
	v.v = u.v + A->d[i].v;
	if(bia::Scene16::IsValidVoxel(grad,v)){
	  q = bia::Scene16::GetVoxelAddress(grad, v);

	  if(bin->data[q]==0) continue;
	  if(Q->L.elem[q].color != BLACK){
	    edge = (ushort)((((int)grad->data[p]) + 
			     ((int)grad->data[q]))>>1);
	    tmp = edge;
	    if(tmp < cost->data[q]){
	      if(Q->L.elem[q].color == GRAY)
		bia::PQueue16::FastRemoveElem(Q,q);
	      // color of q = BLACK
	      cost->data[q] = tmp;
	      pred->data[q] = p;
	      bia::PQueue16::FastInsertElem(Q,q);
	    }
	  }
	}
      }
    }
  }
  bia::AdjRel3::Destroy(&A);
  bia::PQueue16::Destroy(&Q);
  bia::Scene16::Destroy(&cost);

  return pred;
}


void   ResumeFromScratchDIFT_Prototype(bia::Scene16::Scene16 *grad, 
				       Scene *label,
				       bia::Scene16::Scene16 *cost, 
				       Scene *pred, 
				       Scene *mark, Set **seedSet){
  Scene *mst=NULL;
  BMap *bmap=NULL,*isseed=NULL;
  int p,q,n,l1,l2,id=1;
  Set *S=NULL;

  SetScene(mark, 0);
  SetScene(pred, NIL);
  bia::Scene16::Fill(cost, USHRT_MAX);

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



void   ResumeFromScratchDIFT_MinSeeds(Scene *scn,
				      bia::Scene16::Scene16 *grad, 
				      Scene *label,
				      bia::Scene16::Scene16 *cost, 
				      Scene *pred, 
				      Scene *mark, Set **seedSet){
  bia::Scene16::Scene16 *energy;
  Scene *bin,*bkg,*cores,*tmp;
  Set *Se = NULL;
  bia::AdjRel3::AdjRel3 *A;
  AdjRel3 *A_;
  int l,Lmax,p,n;
  int Cmax,Mmax=0;

  grad->dx = scn->dx;
  grad->dy = scn->dy;
  grad->dz = scn->dz;

  SetScene(mark, 0);
  tmp    = CreateScene(grad->xsize, grad->ysize, grad->zsize);
  energy = bia::Scene16::Create(grad->xsize, 
				grad->ysize, 
				grad->zsize);
  n  = grad->n;
  A  = bia::AdjRel3::Spheric(1.0);
  A_ = Spheric(1.0);
  Lmax = MaximumValue3(label);
  for(l=0; l<=Lmax; l++){
    bin = Threshold3(label, l, l);
    bkg = Complement3(bin);
    Se  = Mask2Marker3(bkg, A_);

    SetScene(pred, NIL);
    for(p=0; p<n; p++){
      if(bin->data[p]>0) energy->data[p] = USHRT_MAX;
      else               energy->data[p] = 0;
    }

    RunIFT(grad, energy,
	   pred, tmp, Se);

    cores = GetIFTCores(bin, grad, energy, A);
    //printf("GetCores: %d\n",MaximumValue3(cores));

    //RefineIFTCores(cores, grad, energy, A);
    //printf("RefCores: %d\n",MaximumValue3(cores));

    RemoveRedundantIFTCores(cores, grad, energy, *seedSet, A, 0.3);
    //printf("RemCores: %d\n",MaximumValue3(cores));

    Cmax = MaximumValue3(cores);
    for(p=0; p<n; p++){
      if(cores->data[p]>0)
	mark->data[p] = cores->data[p] + Mmax;
    }
    Mmax += Cmax;
    mark->maxval = Mmax;

    SelectSeedsInIFTCores(scn,grad,cores,seedSet);

    DestroyScene(&cores);
    DestroyScene(&bin);
    DestroyScene(&bkg);
    DestroySet(&Se);
  }
  bia::Scene16::Destroy(&energy);
  DestroyScene(&tmp);
  bia::AdjRel3::Destroy(&A);
  DestroyAdjRel3(&A_);

  SelectBkgSeedsInUniformGrid(label, grad, mark, seedSet, 3);

  SetScene(pred, NIL);
  bia::Scene16::Fill(cost, USHRT_MAX);

  ResumeIFT(label, grad,
	    cost, pred, 
	    mark, *seedSet);
}


/*
Scene *GetIFTCores(Scene *bin, 
		   bia::Scene16::Scene16 *grad, 
		   bia::Scene16::Scene16 *energy, 
		   bia::AdjRel3::AdjRel3 *A){
  Scene *cores;
  int i,s,p,q,n,coreid=0;
  ushort edge,Es;
  bia::Voxel u,v;
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
    
      u.c.x = bia::Scene16::GetAddressX(grad, p);
      u.c.y = bia::Scene16::GetAddressY(grad, p);
      u.c.z = bia::Scene16::GetAddressZ(grad, p);
      for(i=1; i<A->n; i++){
	v.v = u.v + A->d[i].v;

	if(bia::Scene16::IsValidVoxel(grad,v)){
	  q = bia::Scene16::GetVoxelAddress(grad, v);

	  edge = (ushort)((((int)grad->data[p]) + 
			   ((int)grad->data[q]))>>1);
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
*/

/*
void RefineIFTCores(Scene *cores, 
		    bia::Scene16::Scene16 *grad, 
		    bia::Scene16::Scene16 *energy, 
		    bia::AdjRel3::AdjRel3 *A){
  int Cmax,c,i,p,q,s,n;
  ushort edge,Ep,Eq;
  int cs,cp,cq,coreid=0;
  int *equivalent,*hist,*tmp;
  bia::Voxel u,v;
  FIFOQ *Q;
  bia::BMap::BMap *processed;

  n = grad->n;
  Q = FIFOQNew(n);
  processed = bia::BMap::Create(n);
  Cmax = MaximumValue3(cores);
  equivalent = AllocIntArray(Cmax+1);

  for(c=1; c<=Cmax; c++)
    equivalent[c] = c;

  for(s=0; s<n; s++){
    cs = cores->data[s];
    if(cs==0 || bia::BMap::Get(processed, s)) 
      continue;

    bia::BMap::Set1(processed, s);
    FIFOQPush(Q, s);

    while(!FIFOQEmpty(Q)){
      p = FIFOQPop(Q);
      cp = cores->data[p];
      Ep = energy->data[p];

      u.c.x = bia::Scene16::GetAddressX(grad, p);
      u.c.y = bia::Scene16::GetAddressY(grad, p);
      u.c.z = bia::Scene16::GetAddressZ(grad, p);
      for(i=1; i<A->n; i++){
	v.v = u.v + A->d[i].v;

	if(bia::Scene16::IsValidVoxel(grad,v)){
	  q = bia::Scene16::GetVoxelAddress(grad, v);

	  Eq = energy->data[q];
	  cq = cores->data[q];
	  if(cq==0 || bia::BMap::Get(processed, q)) 
	    continue;
	  
	  edge = (ushort)((((int)grad->data[p]) + 
			   ((int)grad->data[q]))>>1);
	  
	  if(equivalent[cp]!=equivalent[cq]){
	    if(edge<=Ep && edge<=Eq){
	      equivalent[cq] = cs;
	      bia::BMap::Set1(processed, q);
	      FIFOQPush(Q, q);
	    }
	  }
	  else{
	    bia::BMap::Set1(processed, q);
	    FIFOQPush(Q, q);
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
  FIFOQDestroy(Q);
  bia::BMap::Destroy(&processed);
}
*/


Scene *GetIFTCores(Scene *bin, 
		   bia::Scene16::Scene16 *grad, 
		   bia::Scene16::Scene16 *energy, 
		   bia::AdjRel3::AdjRel3 *A){
  Scene *cores;
  int i,s,p,q,n,coreid=0;
  ushort edge,Es,Eq;
  bia::Voxel u,v;
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
    
      u.c.x = bia::Scene16::GetAddressX(grad, p);
      u.c.y = bia::Scene16::GetAddressY(grad, p);
      u.c.z = bia::Scene16::GetAddressZ(grad, p);
      for(i=1; i<A->n; i++){
	v.v = u.v + A->d[i].v;

	if(bia::Scene16::IsValidVoxel(grad,v)){
	  q = bia::Scene16::GetVoxelAddress(grad, v);
	  if(bin->data[q]==0) continue;

	  Eq = energy->data[q];
	  if(Eq!=Es) continue;

	  edge = (ushort)((((int)grad->data[p]) + 
			   ((int)grad->data[q]))>>1);
	  if(edge<=Es && cores->data[q]==0){
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


void RemoveRedundantIFTCores(Scene *cores, 
			     bia::Scene16::Scene16 *grad, 
			     bia::Scene16::Scene16 *energy, 
			     Set *seedSet,
			     bia::AdjRel3::AdjRel3 *A,
			     float nvol){
  int Cmax,c,i,p,q,n;
  int cp,cq,coreid=0,total;
  float nh;
  ushort edge,Ep,Eq;
  int *redundant,*tmp,*hist;
  int *marked=NULL;
  Set *S=NULL;
  bia::Voxel u,v;

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

  //Large cores with normalized volume above the 
  //given percentage should be retained:
  hist = AllocIntArray(Cmax+1);
  for(p=0; p<n; p++)
    hist[cores->data[p]]++;
  total = 0;
  for(c=1; c<=Cmax; c++)
    total += hist[c];

  for(c=1; c<=Cmax; c++){
    nh = ((float)hist[c])/((float)total);
    if(nh>nvol) marked[c] = true;
  }
  free(hist);

  for(p=0; p<n; p++){
    Ep = energy->data[p];
    cp = cores->data[p];
    if(cp==0 || redundant[cp]==true) continue;
    u.c.x = bia::Scene16::GetAddressX(grad, p);
    u.c.y = bia::Scene16::GetAddressY(grad, p);
    u.c.z = bia::Scene16::GetAddressZ(grad, p);
    for(i=1; i<A->n; i++){
      v.v = u.v + A->d[i].v;

      if(bia::Scene16::IsValidVoxel(grad,v)){
	q = bia::Scene16::GetVoxelAddress(grad, v);
	cq = cores->data[q];
	Eq = energy->data[q];
	edge = (ushort)((((int)grad->data[p]) + 
			 ((int)grad->data[q]))>>1);
	if(cp!=cq && cq>0){
	  if(edge<=Ep && Ep>=Eq){
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


int _MoveSeedToLowerGradient(bia::Scene16::Scene16 *grad, 
			     Scene *cores, int p, 
			     bia::AdjRel3::AdjRel3 *A){
  bia::Voxel u,v;
  int c,i,q,Gq,Gmin,Gpos;

  c = cores->data[p];
  Gmin = grad->data[p];
  Gpos = p;
  u.c.x = bia::Scene16::GetAddressX(grad, p);
  u.c.y = bia::Scene16::GetAddressY(grad, p);
  u.c.z = bia::Scene16::GetAddressZ(grad, p);
  for(i=1; i<A->n; i++){
    v.v = u.v + A->d[i].v;
    if(bia::Scene16::IsValidVoxel(grad,v)){
      q = bia::Scene16::GetVoxelAddress(grad, v);
      Gq = grad->data[q];
      if(cores->data[q]==c && Gq<Gmin){
	Gpos = q;
	Gmin = Gq;
      }
    }
  }
  return Gpos;
}


void SelectBkgSeedsInUniformGrid(Scene *label,
				 bia::Scene16::Scene16 *grad, 
				 Scene *mark,
				 Set **seedSet,
				 int ndiv){
  float Dx,Dy,Dz;
  int i,j,k,p,nsamples=0;
  bia::Voxel u;
  bia::AdjRel3::AdjRel3 *A;

  Dx = ((float)grad->xsize)/(float)(ndiv+1);
  Dy = ((float)grad->ysize)/(float)(ndiv+1);
  Dz = ((float)grad->zsize)/(float)(ndiv+1);  

  A = bia::AdjRel3::Spheric(1.5);
  bia::AdjRel3::Scale(A, 4./grad->dx, 4./grad->dy, 4./grad->dz);

  for(k=1; k<=ndiv; k++){
    for(i=1; i<=ndiv; i++){
      for(j=1; j<=ndiv; j++){
	u.c.x = ROUND(Dx*j);
	u.c.y = ROUND(Dy*i);
	u.c.z = ROUND(Dz*k);
	if(bia::Scene16::IsValidVoxel(grad,u)){
	  p = bia::Scene16::GetVoxelAddress(grad, u);
	  if(label->data[p]==0){
	    p = _MoveSeedToLowerGradient(grad, label, p, A);
	    InsertSet(seedSet, p);
	    nsamples++;
	    mark->data[p] = mark->maxval + nsamples;
	  }
	}
      }
    }
  }
  mark->maxval += nsamples;
  bia::AdjRel3::Destroy(&A);
}


//Take a seed for each core that has not yet been selected.
void SelectSeedsInIFTCores(Scene *scn, bia::Scene16::Scene16 *grad,
			   Scene *cores, Set **seedSet){
  int Cmax,c,n,p;
  int *marked=NULL;
  Set *S=NULL;
  Voxel *C=NULL; //Central voxel of each core.
  int   *N=NULL; //Number of voxels in each core.
  int   *D=NULL; //Best distance to central node.
  int   *P=NULL; //Closest voxel to the center. 
  int dx,dy,dz,d;
  bia::AdjRel3::AdjRel3 *A;
  
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
      D[c] = 0;
      P[c] = p;
    }
  }
  for(p=0; p<n; p++){
    c = cores->data[p];
    if(c==0 || marked[c] || D[c]==0) continue;
    dx = abs(C[c].x - VoxelX(cores, p));
    dy = abs(C[c].y - VoxelY(cores, p));
    dz = abs(C[c].z - VoxelZ(cores, p));
    d = MAX(MAX(dx,dy),dz);
    if(d<D[c]){
      D[c] = d;
      P[c] = p;
    }
  }

  A = bia::AdjRel3::Spheric(1.5);
  bia::AdjRel3::Scale(A, 4./scn->dx, 4./scn->dy, 4./scn->dz);
  for(c=1; c<=Cmax; c++){
    if(marked[c] || N[c]==0) continue;
    if(P[c]!=NIL){
      P[c] = _MoveSeedToLowerGradient(grad, cores, P[c], A);
      InsertSet(seedSet, P[c]);
      marked[c] = true;
    }
  }
  bia::AdjRel3::Destroy(&A);
  free(P);
  free(D);
  free(C);
  free(N);
  free(marked);
}


void   ResumeIFT(Scene *label, 
		 bia::Scene16::Scene16 *grad,
		 bia::Scene16::Scene16 *cost, 
		 Scene *pred, 
		 Scene *mark, Set *seedSet){
  bia::PQueue16::PQueue16 *Q=NULL;
  int i,p,q,n;
  bia::Voxel u,v;
  ushort edge,tmp,Imax;
  bia::AdjRel3::AdjRel3 *A;
  Set *S;

  SetScene(pred, NIL);
  n    = grad->n;
  Imax = bia::Scene16::GetMaximumValue(grad);
  Q    = bia::PQueue16::Create(Imax+1, //Imax*2+1,
			       n,cost->data);
  Q->C.minvalue = 0;
  A = bia::AdjRel3::Spheric(1.0);

  S = seedSet;
  while(S != NULL){
    p = S->elem;
    cost->data[p] = 0;
    pred->data[p] = NIL;
    bia::PQueue16::FastInsertElem(Q,p);
    S = S->next;
  }

  while(!bia::PQueue16::IsEmpty(Q)){
    p = bia::PQueue16::FastRemoveMinFIFO(Q);
    u.c.x = bia::Scene16::GetAddressX(grad, p);
    u.c.y = bia::Scene16::GetAddressY(grad, p);
    u.c.z = bia::Scene16::GetAddressZ(grad, p);

    for(i=1; i<A->n; i++){
      v.v = u.v + A->d[i].v;

      if(bia::Scene16::IsValidVoxel(grad,v)){
	q = bia::Scene16::GetVoxelAddress(grad, v);
	if(Q->L.elem[q].color != BLACK &&
	   label->data[q]==label->data[p]){
	  
	  edge = (ushort)((((int)grad->data[p]) + 
			   ((int)grad->data[q]))>>1);
	  tmp = MAX(cost->data[p],edge);
	  if(tmp < cost->data[q]){
	    if(Q->L.elem[q].color == GRAY)
	      bia::PQueue16::FastRemoveElem(Q,q);
	    // color of q = BLACK
	    cost->data[q] = tmp;
	    pred->data[q] = p;
	    mark->data[q] = mark->data[p];
	    bia::PQueue16::FastInsertElem(Q,q);
	  }
	}
      }
    }
  }
  bia::AdjRel3::Destroy(&A);
  bia::PQueue16::Destroy(&Q);
}


