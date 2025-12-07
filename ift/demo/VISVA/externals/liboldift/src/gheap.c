#include "gheap.h"


/* Heap Functions */

void GoUpGHeap(GHeap *H, int i) {
  int j = GHEAP_DAD(i);

  while ((i > 0) && (H->compare(H->cost[H->pixel[j]], H->cost[H->pixel[i]]) > 0)) {
    Change(&H->pixel[j], &H->pixel[i]);
    H->pos[H->pixel[i]] = i;
    H->pos[H->pixel[j]] = j;
    i = j;
    j = GHEAP_DAD(i);
  }
}

void GoDownGHeap(GHeap *H, int i) {
  int least, left = GHEAP_LEFTSON(i), right = GHEAP_RIGHTSON(i);

  if ((left <= H->last) && (H->compare(H->cost[H->pixel[left]], H->cost[H->pixel[i]]) < 0))
    least = left;
  else
    least = i;

  if ((right <= H->last) && (H->compare(H->cost[H->pixel[right]], H->cost[H->pixel[least]]) < 0))
    least = right;

  if (least != i) {
    Change(&H->pixel[least], &H->pixel[i]);
    H->pos[H->pixel[i]]     = i;
    H->pos[H->pixel[least]] = least;
    GoDownGHeap(H, least);
 }
}

__inline__ bool IsFullGHeap(GHeap *H) {
  if (H->last == (H->n - 1))
    return true;
  else
    return false;
}

__inline__ bool IsEmptyGHeap(GHeap *H) {
  if (H->last == -1)
    return true;
  else
    return false;
}

GHeap *CreateGHeap(int n, void **cost, int (*compare) (void   *, void   *)) {
  GHeap *H = NULL;
  int i;

  if (cost == NULL) {
    Warning("Cannot create heap without cost map","CreateGHeap");
    return NULL;
  }
  if (compare == NULL) {
    Warning("Cannot create heap without compare function","CreateGHeap");
    return NULL;
  }

  H = (GHeap *) malloc(sizeof(GHeap));
  if (H != NULL) {
    H->n       = n;
    H->cost    = cost;
    H->color   = (char *) malloc(sizeof(char) * n);
    H->pixel   = (int *) malloc(sizeof(int) * n);
    H->pos     = (int *) malloc(sizeof(int) * n);
    H->last    = -1;
    H->compare = compare;
    if (H->color == NULL || H->pos == NULL || H->pixel == NULL)
      Error(MSG1,"CreateGHeap");
    for (i = 0; i < H->n; i++) {
      H->color[i] = WHITE;
      H->pos[i]   = -1;
      H->pixel[i] = -1;
    }    
  } 
  else
    Error(MSG1,"CreateGHeap");
  
  return H;
}

void DestroyGHeap(GHeap **H) {
  GHeap *aux = *H;
  if (aux != NULL) {
    if (aux->pixel != NULL) free(aux->pixel);
    if (aux->color != NULL) free(aux->color);
    if (aux->pos != NULL)   free(aux->pos);
    free(aux);
    *H = NULL;
  }
}

bool InsertGHeap(GHeap *H, int pixel) {
  if (!IsFullGHeap(H)) {
    H->last++;
    H->pixel[H->last] = pixel;
    H->color[pixel]   = GRAY;
    H->pos[pixel]     = H->last;
    GoUpGHeap(H, H->last); 
    return true;
  } else 
    return false;
}

bool RemoveGHeap(GHeap *H, int *pixel) {
  if (!IsEmptyGHeap(H)) {
    *pixel = H->pixel[0];
    H->pos[*pixel]   = -1;
    H->color[*pixel] = BLACK;
    H->pixel[0]      = H->pixel[H->last];
    H->pos[H->pixel[0]] = 0;
    H->last--;
    GoDownGHeap(H, 0);
    return true;
  } else 
    return false;
}

