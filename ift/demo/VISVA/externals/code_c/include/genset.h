#ifndef _GENSET_H_
#define _GENSET_H_

#include "common.h"

typedef struct _genset {
  void *elem;
  struct _genset *next;
} GenSet;

void  InsertGenSet(GenSet **S, void *elem);
void *RemoveGenSet(GenSet **S);
void  DestroyGenSet(GenSet **S);
bool  IsInGenSet(GenSet *S, void *elem);

#endif

