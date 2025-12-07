#ifndef _GLIST_H_
#define _GLIST_H_

#include <stdio.h>
#include <stdlib.h>


typedef struct GListNode
{
    void* data;
    struct GListNode * next;
} GListNode;


/** General purpose list**/
typedef struct GList
{
   GListNode * first;
   GListNode * last;
   void (*freeFunction)(void**);
} GList;

GList* CreateGList (void (*freeF)(void**));

/** Inserts element in list qp. If qp is null
 *  nothing happens.
 **/
void GListInsert (GList * qp, void* t);

/** Frees list. If *s is NULL returns.
 *
 **/
void DestroyGList(GList** s);

#endif
