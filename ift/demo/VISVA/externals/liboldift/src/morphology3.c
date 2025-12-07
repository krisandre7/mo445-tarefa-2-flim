#include "morphology3.h"
#include "geometry.h"
#include "queue.h"
#include "stackandfifo.h"

Scene *Dilate3(Scene *scn, AdjRel3 *A)
{
  Scene *dil = NULL;
  Voxel u, v;
  int p, q, max, i;

  dil = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  dil->dx = scn->dx;
  dil->dy = scn->dy;
  dil->dz = scn->dz;

  for (u.z = 0, p = 0; u.z < scn->zsize; u.z++)
    for (u.y = 0; u.y < scn->ysize; u.y++)
      for (u.x = 0; u.x < scn->xsize; u.x++, p++) {
	max = INT_MIN;
	for (i = 0; i < A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(scn, v.x, v.y, v.z)) {
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    if (scn->data[q] > max)
	      max = scn->data[q];
	  }
	}
	dil->data[p] = max;
      }

  return (dil);
}

Scene *Erode3(Scene *scn, AdjRel3 *A)
{
  Scene *ero = NULL;
  Voxel u, v;
  int p, q, min, i;

  ero = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  ero->dx = scn->dx;
  ero->dy = scn->dy;
  ero->dz = scn->dz;

  for (u.z = 0, p = 0; u.z < scn->zsize; u.z++)
    for (u.y = 0; u.y < scn->ysize; u.y++)
      for (u.x = 0; u.x < scn->xsize; u.x++, p++) {
	min = INT_MAX;
	for (i = 0; i < A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(scn, v.x, v.y, v.z)) {
	    q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    if (scn->data[q] < min)
	      min = scn->data[q];
	  }
	}
	ero->data[p] = min;
      }

  return (ero);
}

Scene *Open3(Scene *scn, AdjRel3 *A)
{
  Scene *open, *ero;

  ero  = Erode3(scn, A);
  open = Dilate3(ero, A);
  DestroyScene(&ero);

  return (open);
}

Scene *Close3(Scene *scn, AdjRel3 *A)
{
  Scene *close, *dil;

  dil   = Dilate3(scn, A);
  close = Erode3(dil, A);
  DestroyScene(&dil);

  return (close);
}



Scene *MorphGrad3(Scene *scn, AdjRel3 *A)
{
  Scene *grad=NULL;
  Voxel u,v;
  int i,s,t,min,max;

  grad = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  grad->dx = scn->dx;
  grad->dy = scn->dy;
  grad->dz = scn->dz;

  for (u.z=0; u.z < scn->zsize; u.z++)
    for (u.y=0; u.y < scn->ysize; u.y++)
      for (u.x=0; u.x < scn->xsize; u.x++){
	s = u.x + scn->tby[u.y] + scn->tbz[u.z];
	min = INT_MAX;
	max = INT_MIN;
	for (i=0; i < A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(scn,v.x,v.y,v.z)){
	    t = v.x + scn->tby[v.y] + scn->tbz[v.z];
	    if (scn->data[t] > max)
	      max = scn->data[t];
	    if (scn->data[t] < min)
	      min = scn->data[t];
	  }
	}
	grad->data[s] = max - min;
      }

  return(grad);
}

Scene *SupRec3(Scene *scn, Scene *marker, AdjRel3 *A)
{
  Scene *cost = NULL;
  Queue *Q = NULL;
  Voxel u, v;
  int i, p, q, n, s, tmp;

  cost = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  cost->dx = scn->dx; cost->dy = scn->dy; cost->dz = scn->dz;
  s = scn->xsize * scn->ysize;
  n = s * scn->zsize;
  Q = CreateQueue(MaximumValue3(marker)+1, n);

  for(p = 0; p < n; p++) {
    cost->data[p] = marker->data[p];
    InsertQueue(Q, cost->data[p], p);
  }

  while (!EmptyQueue(Q)) {
    p = RemoveQueue(Q);
    i   = (p % s);
    u.x =  i % scn->xsize;
    u.y =  i / scn->xsize;
    u.z = p / s;
    for (i = 1; i < A->n; i++) {
      v.z = u.z + A->dz[i];
      v.y = u.y + A->dy[i];
      v.x = u.x + A->dx[i];
      if (ValidVoxel(scn, v.x, v.y, v.z)) {
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (cost->data[p] < cost->data[q]) {
	  tmp = MAX(cost->data[p], scn->data[q]);
	  if (tmp < cost->data[q]) {
	    UpdateQueue(Q, q, cost->data[q], tmp);
	    cost->data[q] = tmp;
	  }
	}
      }
    }
  }
	  
  DestroyQueue(&Q);

  return (cost);
}
  
Scene *InfRec3(Scene *scn, Scene *marker, AdjRel3 *A)
{
  Scene *cost = NULL;
  Queue *Q = NULL;
  Voxel u, v;
  int i, p, q, n, s, Imax, tmp;

  cost = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  cost->dx = scn->dx; cost->dy = scn->dy; cost->dz = scn->dz;
  s = scn->xsize * scn->ysize;
  n = s * scn->zsize;
  Imax = MaximumValue3(scn);
  Q = CreateQueue(Imax+1, n);

  for(p = 0; p < n; p++) {
    cost->data[p] = marker->data[p];
    InsertQueue(Q, Imax - cost->data[p], p);
  }

  while (!EmptyQueue(Q)) {
    p = RemoveQueue(Q);
    i   = (p % s);
    u.x =  i % scn->xsize;
    u.y =  i / scn->xsize;
    u.z = p / s;
    for (i = 1; i < A->n; i++) {
      v.z = u.z + A->dz[i];
      v.y = u.y + A->dy[i];
      v.x = u.x + A->dx[i];
      if (ValidVoxel(scn, v.x, v.y, v.z)) {
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (cost->data[p] > cost->data[q]) {
	  tmp = MIN(cost->data[p], scn->data[q]);
	  if (tmp > cost->data[q]) {
	    UpdateQueue(Q, q, Imax - cost->data[q], Imax - tmp);
	    cost->data[q] = tmp;
	  }
	}
      }
    }
  }
	  
  DestroyQueue(&Q);

  return (cost);
}


Scene *CloseHoles3(Scene *scn)
{
  AdjRel3 *A = NULL;
  Scene *cost = NULL;
  Queue *Q = NULL;
  Voxel u, v;
  int i, p, q, n, s, Imax, tmp;

  cost = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  cost->dx = scn->dx; cost->dy = scn->dy; cost->dz = scn->dz;
  s = scn->xsize * scn->ysize;
  n = s * scn->zsize;
  Imax = MaximumValue3(scn);
  Q = CreateQueue(Imax+2, n);
  A = Spheric(1.0);

  for(p = 0; p < n; p++) {
    if (EdgeVoxel(scn,p,s)) {
      cost->data[p] = scn->data[p];
      InsertQueue(Q, cost->data[p], p);
    } else
      cost->data[p] = Imax+1;
  }

  while (!EmptyQueue(Q)) {
    p = RemoveQueue(Q);
    i   = (p % s);
    u.x = i % scn->xsize;
    u.y = i / scn->xsize;
    u.z = p / s;
    for (i = 1; i < A->n; i++) {
      v.z = u.z + A->dz[i];
      v.y = u.y + A->dy[i];
      v.x = u.x + A->dx[i];
      if (ValidVoxel(scn, v.x, v.y, v.z)) {
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (cost->data[p] < cost->data[q]) {
	  tmp = MAX(cost->data[p], scn->data[q]);
	  if (tmp < cost->data[q]) {
	    InsertQueue(Q, tmp, q);
	    cost->data[q] = tmp;
	  }
	}
      }
    }
  }
	  
  DestroyQueue(&Q);
  DestroyAdjRel3(&A);

  return (cost);
}

Scene *RemDomes3(Scene *scn)
{
  AdjRel3 *A = NULL;
  Scene *cost = NULL;
  Queue *Q = NULL;
  Voxel u, v;
  int i, p, q, n, s, Imin, Imax, tmp;

  cost = CreateScene(scn->xsize, scn->ysize, scn->zsize);
  cost->dx = scn->dx; cost->dy = scn->dy; cost->dz = scn->dz;
  s = scn->xsize * scn->ysize;
  n = s * scn->zsize;
  Imax = MaximumValue3(scn);
  Imin = MinimumValue3(scn);
  Q = CreateQueue(Imax+1, n);
  A = Spheric(1.0);

  for(p = 0; p < n; p++) {
    if (EdgeVoxel(scn,p,s)){
      cost->data[p] = scn->data[p];
      InsertQueue(Q, Imax - cost->data[p], p);
    } else
      cost->data[p] = MAX(Imin-1,0);
  }

  while (!EmptyQueue(Q)) {
    p = RemoveQueue(Q);
    u.z = p / s;
    i = (p % s);
    u.y = i / scn->xsize;
    u.x = i % scn->xsize;
    for (i = 1; i < A->n; i++) {
      v.z = u.z + A->dz[i];
      v.y = u.y + A->dy[i];
      v.x = u.x + A->dx[i];
      if (ValidVoxel(scn, v.x, v.y, v.z)) {
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (cost->data[p] > cost->data[q]) {
	  tmp = MIN(cost->data[p], scn->data[q]);
	  if (tmp > cost->data[q]) {
	    InsertQueue(Q, Imax - tmp, q);
	    cost->data[q] = tmp;
	  }
	}
      }
    }
  }
	  
  DestroyQueue(&Q);
  DestroyAdjRel3(&A);

  return (cost);
}

Scene *OpenRec3(Scene *scn, AdjRel3 *A)
{
  Scene *open, *orec;
  AdjRel3 *A6;

  open = Open3(scn, A);
  A6 = Spheric(1.0);
  orec = InfRec3(scn, open, A6);
  DestroyScene(&open);
  DestroyAdjRel3(&A6);

  return (orec);
}

Scene *CloseRec3(Scene *scn, AdjRel3 *A)
{
  Scene *close, *crec;
  AdjRel3 *A6;

  close = Close3(scn, A);
  A6 = Spheric(1.0);
  crec = SupRec3(scn, close, A6);
  DestroyScene(&close);
  DestroyAdjRel3(&A6);

  return (crec);
}

Scene *Leveling3(Scene *scn1, Scene *scn2)
{
  AdjRel3 *A=NULL;
  Scene *dil,*ero,*and,*or,*infrec,*suprec;

  A      = Spheric(1.0);
  dil    = Dilate3(scn2,A);
  and    = And3(scn1,dil);
  DestroyScene(&dil);
  infrec = InfRec3(scn1,and,A); /* and <= infrec <= scn1 */
  DestroyScene(&and);

  ero    = Erode3(scn1,A);
  or     = Or3(ero,infrec);    
  DestroyScene(&ero);
  suprec = SupRec3(infrec,or,A);  /* and <= suprec <= or */ 
  DestroyScene(&or);
  DestroyScene(&infrec);
  DestroyAdjRel3(&A);

  return(suprec);
}

void iftBasins3(AnnScn *ascn, AdjRel3 *A)
{
  Queue *Q=NULL;
  int i, p, q, s, tmp, Imax, n;
  Voxel u, v;

  s = ascn->scn->xsize*ascn->scn->ysize;
  n = s*ascn->scn->zsize;
  Imax = MaximumValue3(ascn->scn);
  for (p = 0; p < n; p++)
    if (ascn->cost->data[p] != INT_MAX)
      if (Imax < ascn->cost->data[p])
	Imax = ascn->cost->data[p];

  Q = CreateQueue(Imax+1, n);

  while (ascn->seed != NULL) {
    p = RemoveSet(&(ascn->seed));
    InsertQueue(Q, ascn->cost->data[p], p);
  }

  while (!EmptyQueue(Q)) {
    p = RemoveQueue(Q);
    u.z = p / s;
    i = (p % s);
    u.y = i / ascn->scn->xsize;
    u.x = i % ascn->scn->xsize;
    for (i = 1; i < A->n; i++) {
      v.z = u.z + A->dz[i];
      v.y = u.y + A->dy[i];
      v.x = u.x + A->dx[i];
      if (ValidVoxel(ascn->scn, v.x, v.y, v.z)) {
	q = v.x + ascn->scn->tby[v.y] + ascn->scn->tbz[v.z];
	if ((ascn->cost->data[p] < ascn->cost->data[q]) ||
	    (p == ascn->pred->data[q]))
	  {
	    tmp = MAX(ascn->cost->data[p], ascn->scn->data[q]);
	    if ((tmp < ascn->cost->data[q]) ||
		(p == ascn->pred->data[q]))
	      {
		InsertQueue(Q, tmp, q);
		ascn->cost->data[q] = tmp;
		ascn->pred->data[q] = p;
		ascn->label->data[q] = ascn->label->data[p];
	      }
	  }
      }
    }
  }

  DestroyQueue(&Q);
}

void iftDomes3(AnnScn *ascn, AdjRel3 *A)
{
  Queue *Q=NULL;
  int i, p, q, s, tmp, Imax, n;
  Voxel u, v;

  s = ascn->scn->xsize*ascn->scn->ysize;
  n = s*ascn->scn->zsize;
  Imax = MaximumValue3(ascn->scn);
  
  for (p = 0; p < n; p++)
    if (ascn->cost->data[p] != INT_MAX) {
      if (Imax < ascn->cost->data[p])
	Imax = ascn->cost->data[p];
    } else {
      ascn->cost->data[p] = INT_MIN;
    }
  
  Q = CreateQueue(Imax+1, n);

  while (ascn->seed != NULL) {
    p = RemoveSet(&(ascn->seed));
    InsertQueue(Q, Imax-ascn->cost->data[p], p);
  }

  while (!EmptyQueue(Q)) {
    p = RemoveQueue(Q);
    u.z = p / s;
    i = (p % s);
    u.y = i / ascn->scn->xsize;
    u.x = i % ascn->scn->xsize;
    for (i = 1; i < A->n; i++) {
      v.z = u.z + A->dz[i];
      v.y = u.y + A->dy[i];
      v.x = u.x + A->dx[i];
      if (ValidVoxel(ascn->scn, v.x, v.y, v.z)) {
	q = v.x + ascn->scn->tby[v.y] + ascn->scn->tbz[v.z];
	if ((ascn->cost->data[p] > ascn->cost->data[q]) ||
	    (p == ascn->pred->data[q]))
	  {
	    tmp = MIN(ascn->cost->data[p], ascn->scn->data[q]);
	    if ((tmp > ascn->cost->data[q]) ||
		(p == ascn->pred->data[q]))
	      {
		InsertQueue(Q, Imax-tmp, q);
		ascn->cost->data[q] = tmp;
		ascn->pred->data[q] = p;
		ascn->label->data[q] = ascn->label->data[p];
	      }
	  }
      }
    }
  }

  DestroyQueue(&Q);
}

Scene *AreaClose3(Scene *scn, int thres)
{
  Scene  *area=NULL,*cost=NULL,*level=NULL,*pred=NULL,*root=NULL;           
  AdjRel3 *A=NULL;
  Queue  *Q=NULL;
  int i,p,q,r=0,s,n,Imax,tmp,xysize;
  Voxel u,v;

  A        = Spheric(1.0);
  Imax     = MaximumValue3(scn);
  area     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  pred     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  root     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  level    = CopyScene(scn);
  cost     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n        = scn->xsize*scn->ysize*scn->zsize;
  Q        = CreateQueue(Imax+2,n);
  xysize   = scn->xsize*scn->ysize;

  for (p=0; p < n; p++){
    pred->data[p]  = p;
    root->data[p]  = p;
    cost->data[p]  = scn->data[p]+1;
    InsertQueue(Q,cost->data[p],p);
  }
  
  /* Find level for local superior reconstruction */
  
  while (!EmptyQueue(Q)){
    p=RemoveQueue(Q);

    /* Find and update root pixel, level and area */    

    r  = SeedComp3(root,p);

    if ((area->data[r]<= thres)&&(level->data[r] < scn->data[p]))
      level->data[r] = scn->data[p];
    area->data[r] = area->data[r] + 1;

    /* Visit the adjacent pixels */

    tmp = p%xysize;
    u.x = tmp%scn->xsize;
    u.y = tmp/scn->xsize;
    u.z = p/xysize;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (cost->data[p] < cost->data[q]){
	  tmp = MAX(cost->data[p],scn->data[q]);
	  if (tmp < cost->data[q]){
	    UpdateQueue(Q,q,cost->data[q],tmp);
	    cost->data[q] = tmp;
	    root->data[q] = root->data[p];
	    pred->data[q] = pred->data[p];	    	    
	  }
	}else { /* Merge two basins */
	  if (Q->L.elem[q].color == BLACK){
	    s = SeedComp3(root,q);	  
	    if (r != s) {	
	      if ((area->data[s] <= thres)&&(level->data[s]< scn->data[p]))
		level->data[s] = scn->data[p];
	      
	      if (area->data[r] < area->data[s]){
		tmp = r;
		r   = s;
		s   = tmp;
	      }    
	      root->data[s] = r;	    
	      area->data[r] += area->data[s];
	   
	      if ((pred->data[p]==p)&&(pred->data[q]==q)){
		pred->data[s]  = r;
	      }
	    }
	  }
	}
      }
    }
  }

  /* Compute local superior reconstruction */
  
  ResetQueue(Q);

  for (p=0; p < n; p++) {
    if (pred->data[p]==p){
      InsertQueue(Q,Imax-level->data[p],p);
    }
  }

  while (!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    tmp = p%xysize;
    u.x = tmp%scn->xsize;
    u.y = tmp/scn->xsize;
    u.z = p/xysize;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (level->data[p] > level->data[q])
	  {
	    if (Q->L.elem[q].color == GRAY)
	      UpdateQueue(Q,q,Imax-level->data[q],Imax-level->data[p]);
	    else{
	      InsertQueue(Q,Imax-level->data[p],q);
	    }
	    level->data[q]  = level->data[p];
	  }		  
      }
    }
  }

  DestroyQueue(&Q);
  DestroyAdjRel3(&A);
  DestroyScene(&area);    
  DestroyScene(&cost);    
  DestroyScene(&pred);    
  DestroyScene(&root);    

  return(level);
}

Scene *AreaOpen3(Scene *scn, int thres)
{
  Scene  *area=NULL,*cost=NULL,*level=NULL,*pred=NULL,*root=NULL;           
  AdjRel3 *A=NULL;
  Queue  *Q=NULL;
  int i,p,q,r,s,n,Imax,tmp,xysize;
  Voxel u,v;


  A        = Spheric(1.0);
  Imax     = MaximumValue3(scn);
  area     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  pred     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  root     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  level    = CopyScene(scn);
  cost     = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n        = scn->xsize*scn->ysize*scn->zsize;
  Q        = CreateQueue(Imax+2,n);
  xysize   = scn->xsize*scn->ysize;

  for (p=0; p < n; p++){
    pred->data[p]  = p;
    root->data[p]  = p;
    cost->data[p]  = Imax + 1 - scn->data[p];
    InsertQueue(Q,cost->data[p],p);
  }
  
  /* Find level for local inferior reconstruction */
  
  while (!EmptyQueue(Q)){
    p=RemoveQueue(Q);

    /* Find and update root pixel, level and area */    

    r  = SeedComp3(root,p);
    
    if ((area->data[r]<=thres)&&(level->data[r] > scn->data[p]))
      level->data[r] = scn->data[p];
    area->data[r]++;
  
    /* Visit the adjacent pixels */

    tmp = p%xysize;
    u.x = tmp%scn->xsize;
    u.y = tmp/scn->xsize;
    u.z = p/xysize;

    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (cost->data[p] < cost->data[q]){
	  tmp = MAX(cost->data[p],Imax-scn->data[q]);
	  if (tmp < cost->data[q]){
	    UpdateQueue(Q,q,cost->data[q],tmp);
	    cost->data[q] = tmp;
	    root->data[q] = root->data[p];
	    pred->data[q] = root->data[p];	    	    
	  }
	}else { /* Merge two domes */
	  if (Q->L.elem[q].color == BLACK){
	    s = SeedComp3(root,q);	  
	    if (r != s) {	
	      if ((area->data[s] <= thres)&&(level->data[s]>scn->data[p]))
		level->data[s] = scn->data[p];
	      if (area->data[r] < area->data[s]){
		tmp = r;
		r   = s;
		s   = tmp;
	      }    
	      root->data[s] = r;	    
	      area->data[r] += area->data[s];
	      if ((pred->data[p]==p)&&(pred->data[q]==q)){
		pred->data[s]  = r;
	      }
	    }
	  }
	}
      }
    }
  }
     
  /* Compute local inferior reconstruction */
 
  ResetQueue(Q);
  for (p=0; p < n; p++) 
    if (pred->data[p]==p){
      InsertQueue(Q,level->data[p],p);
    }
  
  while (!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    tmp = p%xysize;
    u.x = tmp%scn->xsize;
    u.y = tmp/scn->xsize;
    u.z = p/xysize;
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){
	q = v.x + scn->tby[v.y] + scn->tbz[v.z];
	if (level->data[p] < level->data[q])
	  {
	    if (Q->L.elem[q].color==GRAY)
	      UpdateQueue(Q,q,level->data[q],level->data[p]);
	    else
	      InsertQueue(Q,level->data[p],q);
	    level->data[q]  = level->data[p];
	  }
      }
    }
  }

  DestroyQueue(&Q);
  DestroyAdjRel3(&A);
  DestroyScene(&area);    
  DestroyScene(&cost);    
  DestroyScene(&pred);    
  DestroyScene(&root);    

  return(level);
}

/* It assumes that the next operation is a erosion */

Scene *DilateBin3(Scene *bin, Set **seed, float radius)
{
  Scene *cost=NULL,*root=NULL,*dil=NULL;
  Queue *Q=NULL;
  int i,p,q,n,sz,xysize;
  Voxel u,v,w;
  int *sq=NULL,tmp=INT_MAX,dx,dy,dz;
  float dist;
  AdjRel3 *A=NULL;

  if (*seed == NULL) {
    A      = Spheric(1.0);
    for (u.z=0; u.z < bin->zsize; u.z++) {
      for (u.y=0; u.y < bin->ysize; u.y++){
	for (u.x=0; u.x < bin->xsize; u.x++){
	  p = u.x + bin->tby[u.y]+bin->tbz[u.z];
	  if (bin->data[p]==1){
	    for (i=1; i < A->n; i++){
	      v.x = u.x + A->dx[i];
	      v.y = u.y + A->dy[i];
	      v.z = u.z + A->dz[i];
	      if (ValidVoxel(bin,v.x,v.y,v.z)){
		q = v.x + bin->tby[v.y] + bin->tbz[v.z];
		if (bin->data[q]==0){
		  InsertSet(seed,p);
		  break;
		}
	      }
	    }
	  }
	}
      }
    }
    DestroyAdjRel3(&A);
  }

  dil  = CopyScene(bin);  
  dist = (radius*radius);
  A  = Spheric(1.8);
  n  = MAX(MAX(dil->xsize,dil->ysize),dil->zsize);
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  cost = CreateScene(dil->xsize,dil->ysize,dil->zsize);
  root = CreateScene(dil->xsize,dil->ysize,dil->zsize);
  n    = dil->xsize*dil->ysize*dil->zsize;
  xysize = dil->xsize*dil->ysize;

  sz = FrameSize3(A);
  Q  = CreateQueue(2*sz*(dil->xsize+dil->ysize+dil->zsize)+3*sz*sz,n);

  for (p=0; p < n; p++) 
    cost->data[p]=INT_MAX;

  while (*seed != NULL){
    p=RemoveSet(seed);
    cost->data[p]=0;
    root->data[p]=p;
    InsertQueue(Q,cost->data[p]%Q->C.nbuckets,p);
  }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (cost->data[p] <= dist){

      dil->data[p] = 1;

      u.x = (p % xysize) % cost->xsize;
      u.y = (p % xysize) / cost->xsize;
      u.z = p / xysize;

      w.x = (root->data[p] % xysize) % cost->xsize;
      w.y = (root->data[p] % xysize) / cost->xsize;
      w.z = root->data[p] / xysize;

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	v.z = u.z + A->dz[i];
	if (ValidVoxel(dil,v.x,v.y,v.z)){
	  q = v.x + dil->tby[v.y] + dil->tbz[v.z];
	  if ((cost->data[p] < cost->data[q])&&(dil->data[q]==0)){	   
	    dx  = abs(v.x-w.x);
	    dy  = abs(v.y-w.y);
	    dz  = abs(v.z-w.z);
	    tmp = sq[dx] + sq[dy] + sq[dz];
	    if (tmp < cost->data[q]){
	      if (cost->data[q] == INT_MAX)
		InsertQueue(Q,tmp%Q->C.nbuckets,q);
	      else
		UpdateQueue(Q,q,cost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	      cost->data[q]  = tmp;
	      root->data[q]  = root->data[p];
	    }
	  }
	}
      }
    } else {
      InsertSet(seed,p);
    }
  }

  free(sq);
  DestroyQueue(&Q);
  DestroyScene(&root);
  DestroyScene(&cost);
  DestroyAdjRel3(&A);

  return(dil);
}

/* It assumes that the next operation is a dilation */

Scene *ErodeBin3(Scene *bin, Set **seed, float radius)
{
  Scene *cost=NULL,*root=NULL,*ero=NULL;
  Queue *Q=NULL;
  int i,p,q,n,sz,xysize;
  Voxel u,v,w;
  int *sq=NULL,tmp=INT_MAX,dx,dy,dz;
  float dist;
  AdjRel3 *A=NULL;

  if (*seed == NULL) {
    A      = Spheric(1.0);
    for (u.z=0; u.z < bin->zsize; u.z++) {
      for (u.y=0; u.y < bin->ysize; u.y++){
	for (u.x=0; u.x < bin->xsize; u.x++){
	  p = u.x + bin->tby[u.y]+bin->tbz[u.z];
	  if (bin->data[p]==0){
	    for (i=1; i < A->n; i++){
	      v.x = u.x + A->dx[i];
	      v.y = u.y + A->dy[i];
	      v.z = u.z + A->dz[i];
	      if (ValidVoxel(bin,v.x,v.y,v.z)){
		q = v.x + bin->tby[v.y] + bin->tbz[v.z];
		if (bin->data[q]==1){
		  InsertSet(seed,p);
		  break;
		}
	      }
	    }
	  }
	}
      }
    }
    DestroyAdjRel3(&A);
  }

  ero  = CopyScene(bin);  
  dist = (radius*radius);
  A  = Spheric(1.8);
  n  = MAX(MAX(ero->xsize,ero->ysize),ero->zsize);
  sq = AllocIntArray(n);
  for (i=0; i < n; i++) 
    sq[i]=i*i;

  cost = CreateScene(ero->xsize,ero->ysize,ero->zsize);
  root = CreateScene(ero->xsize,ero->ysize,ero->zsize);
  n    = ero->xsize*ero->ysize*ero->zsize;
  xysize = ero->xsize*ero->ysize;

  sz = FrameSize3(A);
  Q  = CreateQueue(2*sz*(ero->xsize+ero->ysize+ero->zsize)+3*sz*sz,n);

  for (p=0; p < n; p++) 
    cost->data[p]=INT_MAX;

  while (*seed != NULL){
    p=RemoveSet(seed);
    cost->data[p]=0;
    root->data[p]=p;
    InsertQueue(Q,cost->data[p]%Q->C.nbuckets,p);
  }

  while(!EmptyQueue(Q)) {
    p=RemoveQueue(Q);
    if (cost->data[p] <= dist){

      ero->data[p] = 0;

      u.x = (p % xysize) % cost->xsize;
      u.y = (p % xysize) / cost->xsize;
      u.z = p / xysize;

      w.x = (root->data[p] % xysize) % cost->xsize;
      w.y = (root->data[p] % xysize) / cost->xsize;
      w.z = root->data[p] / xysize;

      for (i=1; i < A->n; i++){
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	v.z = u.z + A->dz[i];
	if (ValidVoxel(ero,v.x,v.y,v.z)){
	  q = v.x + ero->tby[v.y] + ero->tbz[v.z];
	  if ((cost->data[p] < cost->data[q])&&(ero->data[q]==1)){	   
	    dx  = abs(v.x-w.x);
	    dy  = abs(v.y-w.y);
	    dz  = abs(v.z-w.z);
	    tmp = sq[dx] + sq[dy] + sq[dz];
	    if (tmp < cost->data[q]){
	      if (cost->data[q] == INT_MAX)
		InsertQueue(Q,tmp%Q->C.nbuckets,q);
	      else
		UpdateQueue(Q,q,cost->data[q]%Q->C.nbuckets,tmp%Q->C.nbuckets);
	      cost->data[q]  = tmp;
	      root->data[q]  = root->data[p];
	    }
	  }
	}
      }
    } else {
      InsertSet(seed,p);
    }
  }

  free(sq);
  DestroyQueue(&Q);
  DestroyScene(&root);
  DestroyScene(&cost);
  DestroyAdjRel3(&A);

  return(ero);
}

Scene *CloseBin3(Scene *bin, float radius)
{
  Scene *close=NULL,*dil=NULL;
  Set *seed=NULL;

  dil   = DilateBin3(bin,&seed,radius);
  close = ErodeBin3(dil,&seed,radius);
  DestroyScene(&dil);
  DestroySet(&seed);

  return(close);
}

Scene *OpenBin3(Scene *bin, float radius)
{
  Scene *open=NULL,*ero=NULL;
  Set *seed=NULL;

  ero   = ErodeBin3(bin,&seed,radius);
  open  = DilateBin3(ero,&seed,radius);
  DestroyScene(&ero);
  DestroySet(&seed);

  return(open);
}

Scene *CloseBinRec3(Scene *bin, float radius)
{
  Scene *close, *crec;
  AdjRel3 *A6;

  close = CloseBin3(bin, radius);
  A6 = Spheric(1.0);
  crec = SupRec3(bin, close, A6);
  DestroyScene(&close);
  DestroyAdjRel3(&A6);

  return (crec);
}

Scene *OpenBinRec3(Scene *bin, float radius)
{
  Scene *open, *orec;
  AdjRel3 *A6;

  open = OpenBin3(bin, radius);
  A6 = Spheric(1.0);
  orec = InfRec3(bin, open, A6);
  DestroyScene(&open);
  DestroyAdjRel3(&A6);

  return (orec);
}

Scene *AsfOCBin3(Scene *bin, float radius)
{
  Scene *dil=NULL,*ero=NULL;
  Set *seed=NULL;

  ero = ErodeBin3(bin,&seed,radius);
  dil = DilateBin3(ero,&seed,2.0*radius);
  DestroyScene(&ero);
  ero = ErodeBin3(dil,&seed,radius);
  DestroyScene(&dil);
  DestroySet(&seed);

  return(ero);
}

Scene *AsfCOBin3(Scene *bin, float radius)
{
  Scene *dil=NULL,*ero=NULL;
  Set *seed=NULL;

  dil = DilateBin3(bin,&seed,radius);
  ero = ErodeBin3(dil,&seed,2.0*radius);
  DestroyScene(&dil);
  dil = DilateBin3(ero,&seed,radius);
  DestroyScene(&ero);
  DestroySet(&seed);

  return(dil);
}

Scene *AsfCOBinRec3(Scene *bin, float radius)
{
  Scene *close, *crec;
  AdjRel3 *A6;

  close = AsfCOBin3(bin, radius);
  A6 = Spheric(1.0);
  crec = SupRec3(bin, close, A6);
  DestroyScene(&close);
  DestroyAdjRel3(&A6);

  return (crec);
}

Scene *AsfOCBinRec3(Scene *bin, float radius)
{
  Scene *open, *orec;
  AdjRel3 *A6;

  open = AsfOCBin3(bin, radius);
  A6 = Spheric(1.0);
  orec = InfRec3(bin, open, A6);
  DestroyScene(&open);
  DestroyAdjRel3(&A6);

  return (orec);
}


Scene *AsfOCRec3(Scene *scn, AdjRel3 *A){
  Scene *open=NULL,*close=NULL;

  open  = OpenRec3(scn,A);
  close = CloseRec3(open,A);
  DestroyScene(&open);

  return(close);
}

Scene *AsfCORec3(Scene *scn, AdjRel3 *A){
  Scene *open=NULL,*close=NULL;

  close = CloseRec3(scn,A);
  open  = OpenRec3(close,A);
  DestroyScene(&close);

  return(open);
}

int SortByVolume (  void *a,   void *b) {

  typedef struct _lbelem {
    int pred;
    int area;
    int max;
    int root;
  } LabelElem;

  LabelElem *la, *lb;
  la = (LabelElem *)a;
  lb = (LabelElem *)b;
  return ((la->area < lb->area) ? -1 : (la->area > lb->area) ? 1 : 0);
}

/* Area open for label images */
/* If diff. regions have same labels use LabelComp3 first */

Scene *LabelAreaOpen3(Scene *label, int thres)
{

  typedef struct _lbelem {
    int pred;
    int area;
    int max;
    int root;
  } LabelElem;

  AdjRel3 *A=Spheric(1.0);
  FIFOQ *Q=NULL;
  int i,p,q,n,l, xysize,Lmax,pred, root, rlabel, plabel;
  Voxel u,v;
  LabelElem *lb;
  LabelElem *lbsorted;
  Lmax = MaximumValue3(label)+1;
  
  n = label->xsize * label->ysize * label->zsize;
  xysize = label->xsize * label->ysize;

  lb = (LabelElem*)calloc(Lmax, sizeof(LabelElem));
  lbsorted = (LabelElem*)calloc(Lmax, sizeof(LabelElem));

  for (i=0; i<n; i++) {
    lb[label->data[i]].area++;
  }
  
  for (l=0; l<Lmax; l++) {
    lb[l].pred=NIL;
  }

  for (u.z=0; u.z < label->zsize; u.y++)
    for (u.y=0; u.y < label->ysize; u.y++)
      for (u.x=0; u.x < label->xsize; u.x++) {
	p = u.x + label->tby[u.y] + label->tbz[u.z];
	for (i=1; i<A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(label, v.x, v.y, v.z)) {
	    q = v.x + label->tby[v.y] + label->tbz[v.z];
	    if (label->data[p] != label->data[q] && 
		lb[ label->data[p] ].area <= thres &&
		lb[ label->data[q] ].area > lb[ label->data[p] ].max) {
	      lb[ label->data[p] ].max = lb[ label->data[q] ].area;
	      lb[ label->data[p] ].pred = q;
	      lb[ label->data[p] ].root = p;
	    }
	  }
	} 
      }  
  
  memcpy(lbsorted, lb, Lmax * sizeof(LabelElem));
  qsort(lbsorted, Lmax, sizeof(LabelElem), SortByVolume);

  for (l=0; l<Lmax; l++) {
    root = lbsorted[l].root;
    rlabel = label->data[root];
    if (lb[rlabel].area <= thres) {
      pred = lbsorted[l].pred;
      plabel = label->data[pred];
      // merge areas;
      lb[plabel].area += lb[rlabel].area;
      // flood fill smaller comp.
      Q = FIFOQNew(lb[rlabel].area);  
      FIFOQPush(Q, root);
      label->data[root] = plabel;
      while (!FIFOQEmpty(Q)) {
	p = FIFOQPop(Q);
	u.x = p % label->xsize;
	u.y = p / label->xsize;
	u.z = p / xysize;
	for (i=1; i<A->n; i++) {
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(label, v.x, v.y, v.z)) {
	    q = v.x + label->tby[v.y] + label->tbz[v.z];
	    if (label->data[q] == rlabel) {                
	      FIFOQPush(Q, q);
	      label->data[q] = plabel;
	    }
	  }
	}          
      }
      FIFOQDestroy(Q);
    }
  }       
  free(lb);
  free(lbsorted);
  DestroyAdjRel3(&A);
  return(label);
}
