
#include "priorityqueue.h"


inline int _FindMinBucketPQueue(PriorityQueue *Q);
inline int _FindMaxBucketPQueue(PriorityQueue *Q);
inline int _BucketFIFOPQueue(PriorityQueue *Q, int bucket);
inline int _BucketLIFOPQueue(PriorityQueue *Q, int bucket);

PriorityQueue *CreatePQueue(int nbuckets, int nelems, int *value){
  PriorityQueue *Q=NULL;

  Q = (PriorityQueue *) malloc(1*sizeof(PriorityQueue));
  
  if (Q != NULL) {
    Q->C.first = (int *)malloc((nbuckets+1) * sizeof(int));
    Q->C.last  = (int *)malloc((nbuckets+1) * sizeof(int));
    Q->C.nbuckets = nbuckets;
    if ( (Q->C.first != NULL) && (Q->C.last != NULL) ){
      Q->L.elem = (PQNode *)malloc(nelems*sizeof(PQNode));
      Q->L.nelems = nelems;
      Q->L.value  = value;
      if (Q->L.elem != NULL){
	ResetPQueue(Q);
      } else
	Error(MSG1,"CreatePQueue");
    } else
      Error(MSG1,"CreatePQueue");
  } else 
    Error(MSG1,"CreatePQueue");
  
  return(Q);
}


void DestroyPQueue(PriorityQueue **Q){
  PriorityQueue *aux;

  aux = *Q;
  if (aux != NULL) {
    if (aux->C.first != NULL) free(aux->C.first);
    if (aux->C.last  != NULL) free(aux->C.last);
    if (aux->L.elem  != NULL) free(aux->L.elem);
    free(aux);
    *Q = NULL;
  }
}


PriorityQueue *GrowPQueue(PriorityQueue **Q, int nbuckets){
  PriorityQueue *Q1;
  int i,bucket;

  Q1 = CreatePQueue(nbuckets,(*Q)->L.nelems,(*Q)->L.value);
  Q1->nadded = (*Q)->nadded;
  Q1->C.minvalue  = (*Q)->C.minvalue;
  Q1->C.maxvalue  = (*Q)->C.maxvalue;
  for (i=0; i<(*Q)->C.nbuckets; i++) 
    if ((*Q)->C.first[i]!=NIL){
      bucket = (*Q)->L.value[(*Q)->C.first[i]]%Q1->C.nbuckets;
      Q1->C.first[bucket] = (*Q)->C.first[i];
      Q1->C.last[bucket]  = (*Q)->C.last[i];
    }
  if ((*Q)->C.first[(*Q)->C.nbuckets]!=NIL){
    bucket = Q1->C.nbuckets;
    Q1->C.first[bucket] = (*Q)->C.first[(*Q)->C.nbuckets];
    Q1->C.last[bucket]  = (*Q)->C.last[(*Q)->C.nbuckets];
  }

  for (i=0; i < (*Q)->L.nelems; i++) 
      Q1->L.elem[i] = (*Q)->L.elem[i];

  DestroyPQueue(Q);
  return(Q1);
}


void ResetPQueue(PriorityQueue *Q){
  int i;

  Q->nadded = 0;
  Q->C.minvalue = INT_MAX;
  Q->C.maxvalue = INT_MIN;
  for (i=0; i < Q->C.nbuckets+1; i++)
    Q->C.first[i]=Q->C.last[i]=NIL;
	
  for (i=0; i < Q->L.nelems; i++) {
    Q->L.elem[i].next =  Q->L.elem[i].prev = NIL;
    Q->L.elem[i].color = WHITE;
  }
}



/* Generic version with circular and growing features */

void   InsertElemPQueue(PriorityQueue **Q, int elem){
  int bucket,value,dvalue;

  (*Q)->nadded++;
  value = (*Q)->L.value[elem];
  if(value==INT_MAX || value==INT_MIN)
    bucket=(*Q)->C.nbuckets;
  else{
    if(value < (*Q)->C.minvalue)
      (*Q)->C.minvalue = value;
    if(value > (*Q)->C.maxvalue)
      (*Q)->C.maxvalue = value;

    dvalue = (*Q)->C.maxvalue - (*Q)->C.minvalue;
    if (dvalue > ((*Q)->C.nbuckets-1)){
      (*Q) = GrowPQueue(Q, 2*(dvalue)+1);
      Warning("Doubling queue size","InsertElemPQueue");
    }
    bucket = value%(*Q)->C.nbuckets;
  }
  if ((*Q)->C.first[bucket] == NIL){ 
    (*Q)->C.first[bucket]   = elem;  
    (*Q)->L.elem[elem].prev = NIL;
  }else {
    (*Q)->L.elem[(*Q)->C.last[bucket]].next = elem;
    (*Q)->L.elem[elem].prev = (*Q)->C.last[bucket];
  }
  
  (*Q)->C.last[bucket]     = elem;
  (*Q)->L.elem[elem].next  = NIL;
  (*Q)->L.elem[elem].color = GRAY;
}


void   RemoveElemPQueue(PriorityQueue *Q, int elem){
  int prev,next,bucket;

  if(Q->L.elem[elem].color!=GRAY) return;

  Q->nadded--;
  if ((Q->L.value[elem]==INT_MAX)||(Q->L.value[elem]==INT_MIN))
    bucket = Q->C.nbuckets;
  else
    bucket = Q->L.value[elem]%Q->C.nbuckets;

  prev = Q->L.elem[elem].prev;
  next = Q->L.elem[elem].next;
  
  /* if elem is the first element */
  if (Q->C.first[bucket] == elem) {
    Q->C.first[bucket] = next;
    if (next == NIL) /* elem is also the last one */
      Q->C.last[bucket] = NIL;
    else
      Q->L.elem[next].prev = NIL;
  }
  else{   /* elem is in the middle or it is the last */
    Q->L.elem[prev].next = next;
    if (next == NIL) /* if it is the last */
      Q->C.last[bucket] = prev;
    else 
      Q->L.elem[next].prev = prev;
  }

  Q->L.elem[elem].color = BLACK;
}


void   UpdateElemPQueue(PriorityQueue **Q, int elem, int newvalue){
  RemoveElemPQueue(*Q, elem);
  (*Q)->L.value[elem] = newvalue;
  InsertElemPQueue(Q, elem);
}


inline int _FindMinBucketPQueue(PriorityQueue *Q){
  int current,last;

  current = Q->C.minvalue%Q->C.nbuckets;
  /** moves to next element **/
  if(Q->C.first[current] == NIL){
    last = current;
    
    do{
      current = (current + 1) % (Q->C.nbuckets);
    }while((Q->C.first[current] == NIL) && (current != last));
    
    if(Q->C.first[current] != NIL)
      Q->C.minvalue = Q->L.value[Q->C.first[current]];
    else{
      if(Q->C.first[Q->C.nbuckets] != NIL){
	current = Q->C.nbuckets;
	Q->C.minvalue = Q->L.value[Q->C.first[current]];
      }
      else
	Error("PQueue is empty","_FindMinBucketPQueue");
    }
  }
  return current;
}


inline int _FindMaxBucketPQueue(PriorityQueue *Q){
  int current,last;

  current = Q->C.maxvalue%Q->C.nbuckets;
  /** moves to next element **/
  if(Q->C.first[current] == NIL){
    last = current;
    
    do{
      current--;
      if(current<0) current = Q->C.nbuckets-1;
    }while((Q->C.first[current] == NIL) && (current != last));
    
    if(Q->C.first[current] != NIL)
      Q->C.maxvalue = Q->L.value[Q->C.first[current]];
    else{
      if(Q->C.first[Q->C.nbuckets] != NIL){
	current = Q->C.nbuckets;
	Q->C.maxvalue = Q->L.value[Q->C.first[current]];
      }
      else
	Error("PQueue is empty","_FindMaxBucketPQueue");
    }
  }
  return current;
}


inline int _BucketFIFOPQueue(PriorityQueue *Q, int bucket){
  int elem=NIL, next;

  elem = Q->C.first[bucket];
  next = Q->L.elem[elem].next;
  if(next == NIL) { /* there was a single element in the list */
    Q->C.first[bucket] = Q->C.last[bucket] = NIL;
  }
  else {
    Q->C.first[bucket] = next;
    Q->L.elem[next].prev = NIL;
  }
  Q->L.elem[elem].color = BLACK;
  return elem;
}


inline int _BucketLIFOPQueue(PriorityQueue *Q, int bucket){
  int elem=NIL, prev;

  elem = Q->C.last[bucket];
  prev = Q->L.elem[elem].prev;
  if(prev == NIL){ /* there was a single element in the list */
    Q->C.last[bucket] = Q->C.first[bucket] = NIL;
  }
  else {
    Q->C.last[bucket] = prev;
    Q->L.elem[prev].next = NIL;
  }
  Q->L.elem[elem].color = BLACK;
  return elem;
}


int    RemoveMinFIFOPQueue(PriorityQueue *Q){
  int bucket;
  Q->nadded--;
  bucket = _FindMinBucketPQueue(Q);
  return _BucketFIFOPQueue(Q, bucket);
}


int    RemoveMinLIFOPQueue(PriorityQueue *Q){
  int bucket;
  Q->nadded--;
  bucket = _FindMinBucketPQueue(Q);
  return _BucketLIFOPQueue(Q, bucket);
}


int    RemoveMaxFIFOPQueue(PriorityQueue *Q){
  int bucket;
  Q->nadded--;
  bucket = _FindMaxBucketPQueue(Q);
  return _BucketFIFOPQueue(Q, bucket);
}


int    RemoveMaxLIFOPQueue(PriorityQueue *Q){
  int bucket;
  Q->nadded--;
  bucket = _FindMaxBucketPQueue(Q);
  return _BucketLIFOPQueue(Q, bucket);
}


/* Faster version to be used with watershed when 
   values are in fixed range [0, nbuckets-1]   */
void FastInsertElemPQueue(PriorityQueue *Q, int elem){
  int bucket;

  Q->nadded++;
  bucket = Q->L.value[elem];

  if(bucket < Q->C.minvalue)
    Q->C.minvalue = bucket;
  if(bucket > Q->C.maxvalue)
    Q->C.maxvalue = bucket;

  if(Q->C.first[bucket] == NIL){ 
    Q->C.first[bucket]   = elem;  
    Q->L.elem[elem].prev = NIL;
  }else {
    Q->L.elem[Q->C.last[bucket]].next = elem;
    Q->L.elem[elem].prev = Q->C.last[bucket];
  }
  Q->C.last[bucket]     = elem;
  Q->L.elem[elem].next  = NIL;
  Q->L.elem[elem].color = GRAY;
}


void FastRemoveElemPQueue(PriorityQueue *Q, int elem){
  int prev,next,bucket;

  Q->nadded--;
  bucket = Q->L.value[elem];
  prev = Q->L.elem[elem].prev;
  next = Q->L.elem[elem].next;
  
  /* if elem is the first element */
  if (Q->C.first[bucket] == elem) {
    Q->C.first[bucket] = next;
    if (next == NIL) /* elem is also the last one */
      Q->C.last[bucket] = NIL;
    else
      Q->L.elem[next].prev = NIL;
  }
  else{   /* elem is in the middle or it is the last */
    Q->L.elem[prev].next = next;
    if (next == NIL) /* if it is the last */
      Q->C.last[bucket] = prev;
    else 
      Q->L.elem[next].prev = prev;
  }
  Q->L.elem[elem].color = BLACK;
}


void FastUpdateElemPQueue(PriorityQueue *Q, int elem, int newvalue){
  FastRemoveElemPQueue(Q, elem);
  Q->L.value[elem] = newvalue;
  FastInsertElemPQueue(Q, elem);
}


inline int _FastFindMinBucketPQueue(PriorityQueue *Q){
  int current;

  current = Q->C.minvalue;
  /** moves to next element **/
  if(Q->C.first[current] == NIL){
    do{
      current++;
    }while((current<Q->C.nbuckets)&&(Q->C.first[current] == NIL));
    
    if(current < Q->C.nbuckets)
      Q->C.minvalue = current;
    else
      Error("PQueue is empty","_FastFindMinBucketPQueue");
  }
  return current;
}


inline int _FastFindMaxBucketPQueue(PriorityQueue *Q){
  int current;

  current = Q->C.maxvalue;
  /** moves to next element **/
  if(Q->C.first[current] == NIL){
    do{
      current--;
    }while((current>=0)&&(Q->C.first[current] == NIL));
    
    if(current >= 0)
      Q->C.maxvalue = current;
    else
      Error("PQueue is empty","_FastFindMaxBucketPQueue");
  }
  return current;
}


int    FastRemoveMinFIFOPQueue(PriorityQueue *Q){
  int bucket;
  Q->nadded--;
  bucket = _FastFindMinBucketPQueue(Q);
  return _BucketFIFOPQueue(Q, bucket);
}

int    FastRemoveMinLIFOPQueue(PriorityQueue *Q){
  int bucket;
  Q->nadded--;
  bucket = _FastFindMinBucketPQueue(Q);
  return _BucketLIFOPQueue(Q, bucket);
}

int    FastRemoveMaxFIFOPQueue(PriorityQueue *Q){
  int bucket;
  Q->nadded--;
  bucket = _FastFindMaxBucketPQueue(Q);
  return _BucketFIFOPQueue(Q, bucket);
}

int    FastRemoveMaxLIFOPQueue(PriorityQueue *Q){
  int bucket;
  Q->nadded--;
  bucket = _FastFindMaxBucketPQueue(Q);
  return _BucketLIFOPQueue(Q, bucket);
}


