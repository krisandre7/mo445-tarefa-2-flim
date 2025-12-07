
#include "arraylist.h"

void myfree(void **mem);

void myfree(void **mem){
  if(*mem!=NULL){
    free(*mem);
    *mem=NULL;
  }
}


ArrayList *CreateArrayList(int cap){
  ArrayList *A;
  int i;

  A = (ArrayList *) calloc(1,sizeof(ArrayList));
  if(A == NULL)
    Error(MSG1,"CreateArrayList");

  A->array = (void **) calloc(cap, sizeof(void *));
  if(A->array == NULL)
    Error(MSG1,"CreateArrayList");

  for(i=0; i<cap; i++)
    A->array[i] = NULL;
  A->cap = cap;
  A->n = 0;
  A->clean = myfree;

  return A;
}


void DestroyArrayList(ArrayList **A){
  void (*clean)();
  ArrayList *aux;
  int i;

  aux = *A;
  clean = aux->clean;
  if(aux != NULL){
    if(aux->array != NULL){
      if(clean!=NULL){
	for(i=0; i<aux->n; i++)
	  if(aux->array[i]!=NULL)
	    (*clean)(&aux->array[i]);
      }
      free(aux->array);
    }
    free(aux);
    *A = NULL;
  }
}


void SetArrayListCleanFunc(ArrayList *A,
			   void (*clean)(void**)){
  A->clean = clean;
}


void AddArrayListElement(ArrayList *A, 
			 void *elem){
  int i;

  if(A->n < A->cap){
    A->array[A->n] = elem;
    A->n++;
  }
  else{
    A->cap = ROUND(A->cap*1.25)+1;
    A->array = (void **)realloc(A->array,
				A->cap*sizeof(void *));
    if(A->array == NULL)
      Error(MSG1,"AddArrayListElement");
    for(i=A->n; i<A->cap; i++)
      A->array[i] = NULL;
    AddArrayListElement(A, elem);
  }
}


void *GetArrayListElement(ArrayList *A, 
			  int index){
  if(index<0 || index>=A->n)
    return NULL;
  return A->array[index];
}


void  DelArrayListElement(ArrayList *A, 
			  int index){
  void (*clean)();
  int i;

  if(index<0 || index>=A->n)
    return;

  clean = A->clean;
  (*clean)(&A->array[index]);

  for(i=index+1; i<A->n; i++)
    A->array[i-1] = A->array[i];
  A->n--;
}


void  DelArrayListElement_2(ArrayList *A,
			    void **elem){
  int i;
  for(i=0; i<A->n; i++){
    if(*elem == GetArrayListElement(A, i)){
      DelArrayListElement(A, i);
      *elem=NULL;
    }
  }
}


void  ResizeArrayList(ArrayList *A, int n){
  void (*clean)();
  int n_old,cap_old,i;

  n_old = A->n;
  cap_old = A->cap;

  A->cap = n;
  A->n = MIN(n,A->n);

  clean = A->clean;
  for(i=A->n; i<n_old; i++)
    if(A->array[i]!=NULL)
      (*clean)(&A->array[i]);

  A->array = (void **)realloc(A->array,
			      n*sizeof(void *));
  if(A->array == NULL)
    Error(MSG1,"ResizeArrayList");
  for(i=A->n; i<A->cap; i++)
    A->array[i] = NULL;
}


void  Trim2SizeArrayList(ArrayList *A){
  ResizeArrayList(A, A->n);
}


