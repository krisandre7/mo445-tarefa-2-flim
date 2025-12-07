
#include "methods_addons.h"

/*
void method_IFTSC3(Scene *arcw, Scene *scn,
		   Set *Si, Set *Se, 
		   Scene *label){
  GQueue *Q=NULL;
  int i,p,q,n,xysize;
  int edge,tmp,Imax;
  Scene *cost;
  Voxel u,v;
  AdjRel3 *A;
  Set *seed=NULL;

  cost = CreateScene(arcw->xsize,
		     arcw->ysize,
		     arcw->zsize);
  xysize = arcw->xsize*arcw->ysize;
  n      = arcw->n;
  Imax = MaximumValue3(arcw);
  Q    = CreateGQueue(Imax*4+1,n,cost->data); //Imax*2+1
  A = Spheric(1.0);

  DrawMarkers3(label, Si, 1);
  DrawMarkers3(label, Se, 0);
  for(p=0; p<n; p++){
    if(label->data[p]==NIL) cost->data[p] = INT_MAX;
    else                    cost->data[p] = 0;
  }
  seed = Si;
  while(seed!=NULL){
    InsertGQueue(&Q, seed->elem);
    seed = seed->next;
  }
  seed = Se;
  while(seed!=NULL){
    InsertGQueue(&Q, seed->elem);
    seed = seed->next;
  }

  while(!EmptyGQueue(Q)){
    p = RemoveGQueue(Q);      
    u.x = VoxelX(arcw, p);
    u.y = VoxelY(arcw, p);
    u.z = VoxelZ(arcw, p);

    for(i=1; i<A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if(ValidVoxel(arcw,v.x,v.y,v.z)){
	q = VoxelAddress(arcw, v.x,v.y,v.z );
	if(Q->L.elem[q].color != BLACK){
	  edge = arcw->data[p] + arcw->data[q];
	  
	  //======ORIENTATION=======
	  if(label->data[p]>0){
	    if(scn->data[p]>=scn->data[q])
	      edge *= 2;
	  }
	  else{
	    if(scn->data[p]<=scn->data[q])
	      edge *= 2;
	  }
	  //========================

	  tmp = MAX(cost->data[p],edge);
	  if(tmp < cost->data[q]){
	    if(Q->L.elem[q].color == GRAY)
	      RemoveGQueueElem(Q,q); // color of q = BLACK
	    cost->data[q] = tmp;
	    label->data[q] = label->data[p];
	    InsertGQueue(&Q,q);
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  DestroyScene(&cost);
  DestroyGQueue(&Q);
}
*/

/*
void method_WeightedDistanceTransform3(Scene *arcw, 
				       Set *Si, Set *Se, 
				       Scene *label,
				       int power){
  RealHeap *Q=NULL;
  int k,i,p,q,n,xysize,Imax;
  real edge,w,tmp;
  real *cost=NULL;
  Voxel u,v;
  AdjRel3 *A;
  Set *seed=NULL;

  xysize = arcw->xsize*arcw->ysize;
  n      = arcw->n;
  cost = AllocRealArray(n);
  Imax = MaximumValue3(arcw);
  Q = CreateRealHeap(n, cost);
  SetRemovalPolicyRealHeap(Q, MINVALUE);
  A = Spheric(1.0);

  DrawMarkers3(label, Si, 1);
  DrawMarkers3(label, Se, 0);
  for(p=0; p<n; p++){
    if(label->data[p]==NIL) cost[p] = REAL_MAX;
    else                    cost[p] = 0.0;
  }
  seed = Si;
  while(seed!=NULL){
    InsertRealHeap(Q, seed->elem);
    seed = seed->next;
  }
  seed = Se;
  while(seed!=NULL){
    InsertRealHeap(Q, seed->elem);
    seed = seed->next;
  }

  while(!IsEmptyRealHeap(Q)){
    RemoveRealHeap(Q, &p);
    u.x = VoxelX(arcw, p);
    u.y = VoxelY(arcw, p);
    u.z = VoxelZ(arcw, p);

    for(i=1; i<A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if(ValidVoxel(arcw,v.x,v.y,v.z)){
	q = VoxelAddress(arcw, v.x,v.y,v.z );
	if(Q->color[q] != BLACK){
	  w = (real)arcw->data[p] + (real)arcw->data[q];
	  edge = 1.0;
	  for(k=0; k<power; k++)
	    edge *= w;

	  tmp = cost[p] + edge;
	  if(tmp < cost[q]){
	    UpdateRealHeap(Q, q, tmp);
	    label->data[q] = label->data[p];
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  free(cost);
  DestroyRealHeap(&Q);
}
*/


//----------------------------------------
// Special-purpose optimized code:


IFTSC3AuxiliaryData *CreateIFTSC3AuxiliaryData(bia::Scene16::Scene16 *arcw){
  IFTSC3AuxiliaryData *aux=NULL;
  int Imax;

  aux = (IFTSC3AuxiliaryData *) calloc(1,sizeof(IFTSC3AuxiliaryData));
  if(aux == NULL)
    bia::Error((char *)MSG1,
	       (char *)"CreateIFTSC3AuxiliaryData");

  Imax = arcw->maxval; //SceneImax(arcw);

  aux->cost = bia::Scene16::Create(arcw);
  bia::Scene16::Fill(aux->cost, 0);
  aux->Q = bia::PQueue16::Create(Imax+1,  //Imax*2+1,
				 arcw->n,
				 (aux->cost)->data);
  aux->A = bia::AdjRel3::Spheric(1.0);
  aux->N = bia::AdjRel3::AdjVoxels(aux->A, arcw);
  return aux;
}


void DestroyIFTSC3AuxiliaryData(IFTSC3AuxiliaryData **aux){
  IFTSC3AuxiliaryData *tmp;

  tmp = *aux;
  if(tmp != NULL){
    if(tmp->cost != NULL) bia::Scene16::Destroy(&tmp->cost);
    if(tmp->Q != NULL)    bia::PQueue16::Destroy(&tmp->Q);
    if(tmp->A != NULL)    bia::AdjRel3::Destroy(&tmp->A);
    if(tmp->N != NULL)    bia::AdjRel3::DestroyAdjVxl(&tmp->N);
    free(tmp);
    *aux = NULL;
  }
}


real SeedmapIFTSC3(bia::AdjSeedmap3::AdjSeedmap3 *asmap,
		   int index, int lb,
		   bia::Voxel vx,
		   bia::Scene16::Scene16 *arcw,
		   Scene *scn,
		   bia::Scene8::Scene8 *label,
		   MRI_Info info,
		   IFTSC3AuxiliaryData *aux){
  bia::PQueue16::PQueue16 *Q = aux->Q;
  bia::Scene16::Scene16 *cost = aux->cost;
  bia::AdjRel3::AdjRel3 *A  = aux->A;
  bia::AdjRel3::AdjVxl  *N  = aux->N;
  int n,xysize;
  int o,i,p,q;
  ushort edge,tmp,e_p,e_q;
  bia::AdjRegion3::AdjRegion3 *S=NULL;
  bia::Voxel u,v,dvx;
  real Csum;
  int  Cn;
  int oa=0; //object  acquisitions.
  int pa=0; //penalty acquisitions.
  real penalty;

  Cn = 0;
  Csum = 0.0;
  dvx.v = vx.v + (asmap->disp)->d[index].v;

  xysize = arcw->xsize*arcw->ysize;
  n      = arcw->n;
  Q->C.minvalue = 0;

  if(!bia::AdjRegion3::FitInside(asmap->bkg_border[index],
				 dvx, arcw, 1)){
    bia::AdjRegion3::Draw(asmap->uncertainty[index],
			  cost, dvx, USHRT_MAX);

    for(o=0; o<=1; o++){
      if(o==0) S = asmap->bkg_border[index];
      if(o==1) S = asmap->obj_border[index];
      for(i=0; i<S->n; i++){
	v.v = dvx.v + S->d[i].v;
	if(bia::Scene16::IsValidVoxel(arcw,v)){
	  p = bia::Scene16::GetVoxelAddress(arcw,v);
	  if(o==0) label->data[p] = 0;
	  else     label->data[p] = lb;
	  bia::PQueue16::FastInsertElem(Q,p);
	}
      }
    }

    while(!bia::PQueue16::IsEmpty(Q)){
      p = bia::PQueue16::FastRemoveMinFIFO(Q);
      u.c.x = bia::Scene16::GetAddressX(arcw, p);
      u.c.y = bia::Scene16::GetAddressY(arcw, p);
      u.c.z = bia::Scene16::GetAddressZ(arcw, p);

      e_p = arcw->data[p];
      e_p = e_p>>1;

      for(i=1; i<A->n; i++){
	v.v = u.v + A->d[i].v;
	if(bia::Scene16::IsValidVoxel(arcw,v)){
	  q = bia::Scene16::GetVoxelAddress(arcw, v);

	  //edge = (ushort)((((int)arcw->data[p]) + ((int)arcw->data[q]))/2);
	  e_q = arcw->data[q];
	  e_q = e_q>>1;
	  edge = e_p + e_q;

	  if(Q->L.elem[q].color != BLACK){

	    tmp = MAX(cost->data[p],edge);
	    if(tmp < cost->data[q]){
	      if(Q->L.elem[q].color == GRAY)
		bia::PQueue16::FastRemoveElem(Q,q);
	      // color of q = BLACK
	      cost->data[q] = tmp;
	      label->data[q] = label->data[p];
	      bia::PQueue16::FastInsertElem(Q,q);
	    }
	  }
	  else{ //color==BLACK=>removed
	    if(label->data[q] != label->data[p]){

	      Csum += (real)edge;
	      Cn++;
	    }
	  }
	}
	else{
	  Csum += (real)0.0;
	  Cn++;
	}
      }
      if(label->data[p]>0){
	oa++;
	if(scn->data[p]<=info.Tcsf ||
	   scn->data[p]>=info.Tup) pa++;
      }
    }//while;
  }
  else{ //FitInside
    p = bia::Scene16::GetVoxelAddress(arcw,dvx);

    bia::AdjRegion3::Optimize(asmap->uncertainty[index], arcw);
    bia::AdjRegion3::DrawOpt(asmap->uncertainty[index],
			     cost, p, USHRT_MAX);

    for(o=0; o<=1; o++){
      if(o==0) S = asmap->bkg_border[index];
      if(o==1) S = asmap->obj_border[index];
      bia::AdjRegion3::Optimize(S, arcw);
      for(i=0; i<S->n; i++){
	q = p + S->dp[i];
	if(o==0) label->data[q] = 0; 
	else     label->data[q] = lb;
	bia::PQueue16::FastInsertElem(Q,q);
      }
    }

    while(!bia::PQueue16::IsEmpty(Q)){
      p = bia::PQueue16::FastRemoveMinFIFO(Q);

      e_p = arcw->data[p];
      e_p = e_p>>1;

      for(i=1; i<N->n; i++){
	q = p + N->dp[i];

	//edge = (ushort)((((int)arcw->data[p]) + ((int)arcw->data[q]))/2);
	e_q = arcw->data[q];
	e_q = e_q>>1;
	edge = e_p + e_q;

	if(Q->L.elem[q].color != BLACK){

	  tmp = MAX(cost->data[p],edge);
	  if(tmp < cost->data[q]){
	    if(Q->L.elem[q].color == GRAY)
	      bia::PQueue16::FastRemoveElem(Q,q); 
	    // color of q = BLACK
	    cost->data[q] = tmp;
	    label->data[q] = label->data[p];
	    bia::PQueue16::FastInsertElem(Q,q);
	  }
	}
	else{ //color==BLACK=>removed
	  if(label->data[q] != label->data[p]){

	    Csum += (real)edge;
	    Cn++;
	  }
	}
      }
      if(label->data[p]>0){
	oa++;
	if(scn->data[p]<=info.Tcsf ||
	   scn->data[p]>=info.Tup) pa++;
      }
    }//while;
  }


  // Reset queue manually
  if(!bia::AdjRegion3::FitInside(asmap->bkg_border[index],
				 dvx, arcw, 1)){
    for(o=0; o<=2; o++){
      if(o==0) S = asmap->bkg_border[index];
      if(o==1) S = asmap->obj_border[index];
      if(o==2) S = asmap->uncertainty[index];
      for(i=0; i<S->n; i++){
	v.v = dvx.v + S->d[i].v;
	if(bia::Scene16::IsValidVoxel(arcw,v)){
	  p = bia::Scene16::GetVoxelAddress(arcw,v);
	  cost->data[p] = 0;
	  Q->L.elem[p].color = WHITE;
	}
      }
    }
  }
  else{ //FitInside
    p = bia::Scene16::GetVoxelAddress(arcw,dvx);
    for(o=0; o<=2; o++){
      if(o==0) S = asmap->bkg_border[index];
      if(o==1) S = asmap->obj_border[index];
      if(o==2) S = asmap->uncertainty[index];
      bia::AdjRegion3::Optimize(S, arcw);
      for(i=0; i<S->n; i++){
	q = p + S->dp[i];
	cost->data[q] = 0;
	Q->L.elem[q].color = WHITE;
      }
    }
  }
  Q->C.minvalue = USHRT_MAX;
  Q->C.maxvalue = 0;
  Q->nadded = 0;

  penalty = (1.0-((real)pa)/((real)oa));
  penalty = penalty*penalty;
  penalty = penalty*penalty;
  return (Csum/Cn)*penalty;
}


/*
real SeedmapWeightedDistance3(AdjSeedmap3 *asmap,
			      int l,
			      Voxel vx,
			      Scene *arcw,
			      Scene *label,
			      int power){
  static real *cost=NULL;
  static RealHeap *Q=NULL;
  static int *px=NULL;
  static int prev_pxsize=0;
  static int prev_n=0;
  static AdjRel3 *A=NULL;
  int n,xysize,pxsize,npx;
  int o,i,k,p,q;
  real edge,w,tmp;
  Cloud3 *S=NULL;
  Voxel u,v,dvx;
  real Csum;
  int  Cn;

  Cn = 0;
  Csum = 0.0;
  dvx.x = vx.x + (asmap->disp)->dx[l];
  dvx.y = vx.y + (asmap->disp)->dy[l];
  dvx.z = vx.z + (asmap->disp)->dz[l];
  xysize = arcw->xsize*arcw->ysize;
  n      = arcw->n;

  pxsize  = (asmap->uncertainty[l])->n;
  pxsize += (asmap->obj_border[l])->n;
  pxsize += (asmap->bkg_border[l])->n;

  if(prev_n!=n){
    prev_n = n;
    if(cost!=NULL) free(cost);
    if(Q!=NULL)    DestroyRealHeap(&Q);
    cost = NULL;
    Q    = NULL;
  }
  if(pxsize!=prev_pxsize){
    prev_pxsize = pxsize;
    if(px!=NULL) free(px);
    px = NULL;
  }

  if(cost==NULL){
    cost = AllocRealArray(arcw->n);
    for(p=0; p<arcw->n; p++) cost[p] = 0.0;
  }
  if(Q==NULL){
    Q = CreateRealHeap(n, cost);
    SetRemovalPolicyRealHeap(Q, MINVALUE);
  }
  if(px==NULL) px = AllocIntArray(pxsize);
  if(A==NULL)  A = Spheric(1.0);

  for(o=0; o<=2; o++){
    if(o==0) S = asmap->bkg_border[l];
    if(o==1) S = asmap->obj_border[l];
    if(o==2) S = asmap->uncertainty[l];
    for(i=0; i<S->n; i++){
      v.x = dvx.x + S->dx[i];
      v.y = dvx.y + S->dy[i];
      v.z = dvx.z + S->dz[i];
      if(ValidVoxel(arcw,v.x,v.y,v.z)){
	p = VoxelAddress(arcw,v.x,v.y,v.z);
	if(o==2) cost[p] = REAL_MAX;
	else{
	  if(o==0) label->data[p] = 0;
	  else     label->data[p] = l;
	  InsertRealHeap(Q, p);
	}
      }
    }
  }
  
  npx = 0;
  while(!IsEmptyRealHeap(Q)){
    RemoveRealHeap(Q, &p);
    u.x = VoxelX(arcw, p);
    u.y = VoxelY(arcw, p);
    u.z = VoxelZ(arcw, p);
    px[npx] = p;
    npx++;

    for(i=1; i<A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if(ValidVoxel(arcw,v.x,v.y,v.z)){
	q = VoxelAddress( arcw, v.x,v.y,v.z );
	if(Q->color[q] != BLACK){
	  w = (real)arcw->data[p] + (real)arcw->data[q];
	  edge = 1.0;
	  for(k=0; k<power; k++)
	    edge *= w;

	  tmp = cost[p] + edge;
	  if(tmp < cost[q]){
	    UpdateRealHeap(Q, q, tmp);
	    label->data[q] = label->data[p];
	  }
	}
	else{ //color==BLACK=>removed
	  if(label->data[q] != label->data[p]){
	    edge = (real)(arcw->data[p] + arcw->data[q]);
	    Csum += edge;
	    Cn++;
	  }
	}
      }
      else{
	Csum += (real)0.0;
	Cn++;
      }
    }
  }

  // Reset queue manually
  for(i=0; i<npx; i++){
    p = px[i];
    Q->color[p] = WHITE;
    Q->pos[p]   = -1;
    Q->pixel[p] = -1;
    cost[p]     = 0.0;
  }
  Q->last = -1;

  return (Csum/Cn);
}
*/



