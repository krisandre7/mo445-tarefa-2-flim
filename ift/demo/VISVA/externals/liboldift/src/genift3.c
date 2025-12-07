#include "genift3.h"
#include "heap.h"
#include "queue.h"
#include "scene.h"

int FEucl3(AnnScn *ascn, int s, int t)
{
  int dx,dy,dz,p,xysize=ascn->scn->xsize*ascn->scn->ysize;
  Voxel u,v;
  Scene *scn=ascn->scn;
    
  p   = ascn->label->data[s]%xysize;
  u.x = p%scn->xsize;
  u.y = p/scn->xsize;
  u.z = ascn->label->data[s]/xysize;

  p   = t%xysize;
  v.x = p%scn->xsize;
  v.y = p/scn->xsize;
  v.z = t/xysize;

  dx = v.x-u.x;
  dy = v.y-u.y;
  dz = v.z-u.z;

  return(dx*dx + dy*dy + dz*dz); 
}


int FEuclIn3(AnnScn *ascn, int s, int t)
{
  int dx,dy,dz,p,xysize=ascn->scn->xsize*ascn->scn->ysize;
  Voxel u,v;
  Scene *scn=ascn->scn;

  if (scn->data[t] == 0) return (INT_MAX);

  p   = ascn->label->data[s]%xysize;
  u.x = p%scn->xsize;
  u.y = p/scn->xsize;
  u.z = ascn->label->data[s]/xysize;

  p   = t%xysize;
  v.x = p%scn->xsize;
  v.y = p/scn->xsize;
  v.z = t/xysize;

  dx = v.x-u.x;
  dy = v.y-u.y;
  dz = v.z-u.z;

  return(dx*dx + dy*dy + dz*dz);   
}

int FEuclOut3(AnnScn *ascn, int s, int t)
{
  int dx,dy,dz,p,xysize=ascn->scn->xsize*ascn->scn->ysize;
  Voxel u,v;
  Scene *scn=ascn->scn;

  if (scn->data[t] != 0) return (INT_MAX);

  p   = ascn->label->data[s]%xysize;
  u.x = p%scn->xsize;
  u.y = p/scn->xsize;
  u.z = ascn->label->data[s]/xysize;

  p   = t%xysize;
  v.x = p%scn->xsize;
  v.y = p/scn->xsize;
  v.z = t/xysize;

  dx = v.x-u.x;
  dy = v.y-u.y;
  dz = v.z-u.z;

  return(dx*dx + dy*dy + dz*dz);   
}

void IFT3(AnnScn *ascn, AdjRel3 *A, PathCost3 Pcost, int maxinc)
{
  Queue *Q=NULL;
  //  Heap *H=NULL; 
  int i,s,t,n,cost,p,xysize=ascn->scn->xsize*ascn->scn->ysize;
  Voxel u,v;
  Scene *scn=ascn->scn;

  n = scn->xsize*scn->ysize*scn->zsize;
  //  H = CreateHeap(n,ascn->cost->data);
  Q = CreateQueue(maxinc+1,n);
   
  while (ascn->seed != NULL){
    t=RemoveSet(&(ascn->seed));    
      //  InsertHeap(H,t);
      InsertQueue(Q,ascn->cost->data[t]%Q->C.nbuckets,t);
  }
  //  while(!HeapIsEmpty(H)) {
  while(!EmptyQueue(Q)) {
    //RemoveHeap(H,&s);
    s=RemoveQueue(Q);
    p   = s%xysize;
    u.x = p%scn->xsize;
    u.y = p/scn->xsize;
    u.z = s/xysize;    
    for (i=1; i < A->n; i++){
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      v.z = u.z + A->dz[i];
      if (ValidVoxel(scn,v.x,v.y,v.z)){
	t = v.x + scn->tby[v.y] + scn->tbz[v.z];
	//	if (H->color[t] != BLACK){
	if (ascn->cost->data[s] < ascn->cost->data[t]){
	  cost = Pcost(ascn,s,t); 
	  if (cost < ascn->cost->data[t]){
	    if (ascn->cost->data[t] == INT_MAX)
	      InsertQueue(Q,cost%Q->C.nbuckets,t);
	    else
	      UpdateQueue(Q,t,ascn->cost->data[t]%Q->C.nbuckets,cost%Q->C.nbuckets);
	    
	    ascn->cost->data[t]  = cost;
	    ascn->pred->data[t]  = s;
	    ascn->label->data[t] = ascn->label->data[s];
	    /*	      if (H->color[t] == WHITE)
		      InsertHeap(H,t);
		      else
		      GoUp(H,H->pos[t]); */
	  }
	}
      }
    }
  }
  //  DestroyHeap(&H);
  DestroyQueue(&Q);
}
