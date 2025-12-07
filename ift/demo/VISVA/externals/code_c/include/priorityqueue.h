#ifndef _PRIORITYQUEUE_H_
#define _PRIORITYQUEUE_H_

#include "common.h"


typedef struct _pqnode { 
  int  next;  /* next node */
  int  prev;  /* prev node */
  char color; /* WHITE=0, GRAY=1, BLACK=2 */ 
} PQNode;

typedef struct _pqdoublylinkedlists {
  PQNode *elem;  /* all possible doubly-linked lists of the circular queue */
  int nelems;  /* total number of elements */
  int *value;   /* the value of the nodes in the graph */
} PQDoublyLinkedLists; 

typedef struct _pqcircularqueue { 
  int  *first;   /* list of the first elements of each doubly-linked list */
  int  *last;    /* list of the last  elements of each doubly-linked list  */
  int  nbuckets; /* number of buckets in the circular queue */
  int  minvalue;  /* minimum value of a node in queue */
  int  maxvalue;  /* maximum value of a node in queue */
} PQCircularQueue;

typedef struct _priorityqueue { /* Priority queue by Dial implemented as
                                   proposed by A. Falcao */
  PQCircularQueue C;
  PQDoublyLinkedLists L;
  int nadded;      /* Number of elements added. */
} PriorityQueue;



PriorityQueue *CreatePQueue(int nbuckets, int nelems, int *value);
void           DestroyPQueue(PriorityQueue **Q);
PriorityQueue *GrowPQueue(PriorityQueue **Q, int nbuckets);
void           ResetPQueue(PriorityQueue *Q);
#define        IsEmptyPQueue(q) (q->nadded==0)
#define        IsFullPQueue(q) (q->nadded==(q->L).nelems)


/* Generic version with circular and growing features */
void   InsertElemPQueue(PriorityQueue **Q, int elem);
void   RemoveElemPQueue(PriorityQueue *Q, int elem);
void   UpdateElemPQueue(PriorityQueue **Q, int elem, int newvalue);
int    RemoveMinFIFOPQueue(PriorityQueue *Q);
int    RemoveMinLIFOPQueue(PriorityQueue *Q);
int    RemoveMaxFIFOPQueue(PriorityQueue *Q);
int    RemoveMaxLIFOPQueue(PriorityQueue *Q);


/* Faster version to be used with watershed when 
   values are in fixed range [0, nbuckets-1]   */
void   FastInsertElemPQueue(PriorityQueue *Q, int elem);
void   FastRemoveElemPQueue(PriorityQueue *Q, int elem);
void   FastUpdateElemPQueue(PriorityQueue *Q, int elem, int newvalue);
int    FastRemoveMinFIFOPQueue(PriorityQueue *Q);
int    FastRemoveMinLIFOPQueue(PriorityQueue *Q);
int    FastRemoveMaxFIFOPQueue(PriorityQueue *Q);
int    FastRemoveMaxLIFOPQueue(PriorityQueue *Q);

#endif

