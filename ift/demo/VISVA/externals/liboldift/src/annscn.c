#include "annscn.h"

Scene *CreateMapScene(Scene *mainscn)
{
  Scene *scn=NULL;

  scn = (Scene *) calloc(1,sizeof(Scene));
  if (scn == NULL){
    Error(MSG1,"CreateMapScene");
  }
  
  scn->data  = AllocIntArray(mainscn->xsize*mainscn->ysize*mainscn->zsize);
  scn->xsize = mainscn->xsize;
  scn->ysize = mainscn->ysize;
  scn->zsize = mainscn->zsize;
  scn->dx    = mainscn->dx;
  scn->dy    = mainscn->dy;
  scn->dz    = mainscn->dz;
  scn->tby   = mainscn->tby;
  scn->tbz   = mainscn->tbz;
  
  return(scn);
}

AnnScn *Annotate3(Scene *scn)
{
  AnnScn *ascn=NULL;
  int i,n;

  ascn = (AnnScn *) calloc(1,sizeof(AnnScn));
  if (ascn != NULL) {
    n = scn->xsize*scn->ysize*scn->zsize;
    ascn->scn   = scn;
    ascn->cost  = CreateMapScene(scn);
    ascn->pred  = CreateMapScene(scn);
    ascn->label = CreateMapScene(scn);
    ascn->root  = CreateMapScene(scn); 
    ascn->seed  = NULL;
    ascn->mark  = NULL;
    for (i=0; i < n; i++) {
      ascn->cost->data[i]  = INT_MAX;
      ascn->pred->data[i]  = i;
      ascn->label->data[i] = 0;
    }
  } else {
    Error(MSG1,"AnnScn");
  }

  return(ascn);
}


void DestroyMapScene(Scene *scn)
{
  if(scn != NULL){
    if (scn->data != NULL)  free(scn->data); 
    free(scn);    
  }
}

void   DeAnnotate3(AnnScn **ascn)
{
  AnnScn *aux;

  aux = *ascn;
  if (aux != NULL) {
    if (aux->cost  != NULL) DestroyMapScene(aux->cost);
    if (aux->pred  != NULL) DestroyMapScene(aux->pred);
    if (aux->label != NULL) DestroyMapScene(aux->label);
    if (aux->root  != NULL) DestroyMapScene(aux->root);
    if (aux->seed  != NULL) DestroySet(&(aux->seed));
    if (aux->mark  != NULL) DestroySet(&(aux->mark));
    free(aux);
    *ascn = NULL;
  }  
}


void  AddMark3(AnnScn *ascn, int x, int y, int z)
{
  if (ValidVoxel(ascn->scn,x,y,z))
    InsertSet(&(ascn->mark),x+ascn->scn->tby[y]+ascn->scn->tbz[z]);
}

void  AddSeed3(AnnScn *ascn, int x, int y, int z, int cost, int label, int pred)
{
  int p;

  if (ValidVoxel(ascn->scn,x,y,z)){
    p=x+ascn->scn->tby[y]+ascn->scn->tbz[z];
    if (cost < ascn->cost->data[p]){
      InsertSet(&(ascn->seed),p);
      ascn->cost->data[p]  = cost;
      ascn->label->data[p] = label;
      ascn->pred->data[p]  = pred;
    }
  }
}

int FindRoot(Scene *pred, int t)
{
  while(pred->data[t] != t) 
    t = pred->data[t];
  return(t);
}

Scene *GetLabel3(AnnScn *ascn)
{
  return (ascn->label);
}

Scene *GetCost3(AnnScn *ascn)
{
  return (ascn->cost);
}

Scene *GetPred3(AnnScn *ascn)
{
  return (ascn->pred);
}

int SeedComp3(Scene *pred, int p)
{
  int root;

  if (pred->data[p]==p)
    return(p);
  else {
    root = SeedComp3(pred,pred->data[p]);
    pred->data[p] = root;
    return(root);
  }
}
