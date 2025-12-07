#ifndef _HEAP_H_
#define _HEAP_H_

#include "common.h"

typedef struct _heap {
  int *cost;
  char *color;
  int *pixel;
  int *pos;
  int last;
  int n;
} Heap;


/* Auxiliary Functions */

#define HEAP_DAD(i) ((i - 1) / 2)
#define HEAP_LEFTSON(i) (2 * i + 1)
#define HEAP_RIGHTSON(i) (2 * i + 2)

/* Heap Functions */


//This method is deprecated; you 
//should use "IsFullHeap" instead.
//It is still available only for 
//compatibility purposes.

bool HeapIsFull(Heap *H);


//This method is deprecated; you 
//should use "IsEmptyHeap" instead.
//It is still available only for 
//compatibility purposes.
bool HeapIsEmpty(Heap *H);

bool IsFullHeap(Heap *H);
bool IsEmptyHeap(Heap *H);
Heap *CreateHeap(int n, int *cost);
void DestroyHeap(Heap **H);
bool InsertHeap(Heap *H, int pixel);
bool RemoveHeap(Heap *H, int *pixel);
void GoUpHeap(Heap *H, int i);
void GoDownHeap(Heap *H, int i);
void ResetHeap(Heap *H);

#endif



