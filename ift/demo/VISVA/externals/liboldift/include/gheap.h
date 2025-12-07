#ifndef _GHEAP_H_

#define _GHEAP_H_

#include "common.h"

typedef struct _gheap {
  void **cost;
  char *color;
  int *pixel;
  int *pos;
  int last;
  int n;
  int (*compare) (void   *, void   *);
} GHeap;

/* Auxiliary Functions */

#define GHEAP_DAD(i) ((i - 1) / 2)
#define GHEAP_LEFTSON(i) (2 * i + 1)
#define GHEAP_RIGHTSON(i) (2 * i + 2)

/* Heap Functions */

bool IsFullGHeap(GHeap *H);
bool IsEmptyGHeap(GHeap *H);
GHeap *CreateGHeap(int n, void **cost, int (*compare) (void   *, void   *));
void DestroyGHeap(GHeap **H);
bool InsertGHeap(GHeap *H, int pixel);
bool RemoveGHeap(GHeap *H, int *pixel);
void GoUpGHeap(GHeap *H, int i);
void GoDownGHeap(GHeap *H, int i);

#endif










