
#include "heap.h"


/* Heap Functions */

void GoUpHeap(Heap *H, int i){
  int j = HEAP_DAD(i);

  while((i > 0) && (H->cost[H->pixel[j]] > H->cost[H->pixel[i]])){
    Change(&H->pixel[j],&H->pixel[i]);
    H->pos[H->pixel[i]]=i;
    H->pos[H->pixel[j]]=j;
    i = j;
    j = HEAP_DAD(i);
  }
}

void GoDownHeap(Heap *H, int i){
  int least, left=HEAP_LEFTSON(i), right=HEAP_RIGHTSON(i);

  if ((left <= H->last)&&(H->cost[H->pixel[left]] < H->cost[H->pixel[i]]))
    least = left;
  else
    least = i;

  if ((right <= H->last)&&(H->cost[H->pixel[right]] < H->cost[H->pixel[least]]))
    least = right;

  if (least != i){
    Change(&H->pixel[least],&H->pixel[i]);
    H->pos[H->pixel[i]]=i;
    H->pos[H->pixel[least]]=least;
    GoDownHeap(H,least);
 }

}


//This method is deprecated; you 
//should use "IsFullHeap" instead.
//It is still available only for 
//compatibility purposes.
bool HeapIsFull(Heap *H){
  return IsFullHeap(H);
}

bool IsFullHeap(Heap *H){
  if (H->last == (H->n-1))
    return(true);
  else
    return(false);
}


//This method is deprecated; you 
//should use "IsEmptyHeap" instead.
//It is still available only for 
//compatibility purposes.
bool HeapIsEmpty(Heap *H){
  return IsEmptyHeap(H);
}

bool IsEmptyHeap(Heap *H){
  if (H->last == -1){
    return(true);
  }else
    return(false);
}

Heap *CreateHeap(int n, int *cost){
  Heap *H=NULL;
  int i;

  if (cost == NULL){
    Warning("Cannot create heap without cost map","CreateHeap");
    return(NULL);
  }

  H = (Heap *) calloc(1,sizeof(Heap));
  if (H != NULL) {
    H->n     = n;
    H->cost  = cost;
    H->color = (char *) calloc(n,sizeof(char));
    H->pixel = (int *) calloc(n,sizeof(int));
    H->pos   = (int *) calloc(n,sizeof(int));
    H->last = -1;
    if ((H->color == NULL) || (H->pos == NULL) || (H->pixel == NULL))
      Error(MSG1,"CreateHeap");
    for (i=0; i < H->n; i++){
      H->color[i]=WHITE;
      H->pos[i]=-1;
      H->pixel[i]=-1;
    }    
  } 
  else
    Error(MSG1,"CreateHeap");
  
  return(H);
}

void DestroyHeap(Heap **H){
  Heap *aux;
  
  aux = *H;
  if (aux != NULL) {
    if (aux->pixel != NULL) free(aux->pixel);
    if (aux->color != NULL) free(aux->color);
    if (aux->pos != NULL)   free(aux->pos);
    free(aux);
    *H = NULL;
  }
}

bool InsertHeap(Heap *H, int pixel){
  if (!IsFullHeap(H)){
    H->last ++;
    H->pixel[H->last] = pixel;
    H->color[pixel]   = GRAY;
    H->pos[pixel]     = H->last;
    GoUpHeap(H,H->last); 
    return(true);
  } else 
    return(false);
}

bool RemoveHeap(Heap *H, int *pixel){
  if (!IsEmptyHeap(H)){
    *pixel = H->pixel[0];
    H->pos[*pixel]=-1;
    H->color[*pixel] = BLACK;
    H->pixel[0] = H->pixel[H->last];
    H->pos[H->pixel[0]] = 0;
    H->last--;
    GoDownHeap(H,0);
    return(true);
  } else 
    return(false);
}

void ResetHeap(Heap *H)
{
  int i;

  for (i=0; i < H->n; i++) {
    H->color[i] = WHITE;
    H->pos[i]   = -1;
    H->pixel[i] = -1;
  }
  H->last = -1;
}


