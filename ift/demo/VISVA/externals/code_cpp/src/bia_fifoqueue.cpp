
#include "bia_fifoqueue.h"

namespace bia{
  namespace FIFOQueue{

    FIFOQueue *Create(int nbuckets){
      FIFOQueue *Q;

      Q = (FIFOQueue *) malloc(sizeof(FIFOQueue));
      if(Q==NULL) bia::Error((char *)MSG1,
			     (char *)"FIFOQueue::Create");

      Q->nbuckets = nbuckets;
      Q->nadded = 0;
      Q->put = Q->get = 0;
      Q->data = bia::AllocIntArray(nbuckets);
      return Q;
    }

    void    Destroy(FIFOQueue **Q){
      FIFOQueue *aux;
      aux = *Q;
      if(aux!=NULL){
	if(aux->data!=NULL) bia::FreeIntArray(&aux->data);
	free(aux);
	*Q = NULL;
      }  
    }


    void    Push(FIFOQueue *Q, int p){
      int *new_data=NULL;
      int n = Q->nbuckets;

      if(!IsFull(Q)){
	Q->nadded++;
	Q->data[Q->put] = p;
	Q->put = (Q->put + 1) % Q->nbuckets;
      }
      else{ //Doubling queue size.
	new_data = bia::AllocIntArray(2*n);
	
	if(Q->get>Q->put){
	  memcpy(new_data, &(Q->data[Q->get]), 
		 sizeof(int)*(n-Q->get));
	  memcpy(&(new_data[n-Q->get]), Q->data, 
		 sizeof(int)*(Q->put+1));
	}
	else{
	  memcpy(new_data, &(Q->data[Q->get]), 
		 sizeof(int)*Q->nadded);
	}
	bia::FreeIntArray(&Q->data);
	Q->data = new_data;
	Q->nbuckets = 2*n;
	Q->put = Q->nadded;
	Q->get = 0;
	Push(Q, p);
      }
    }


    // returns NIL if empty.
    int     Pop(FIFOQueue *Q){ 
      int v;
      if(IsEmpty(Q)) return NIL;
      v = Q->data[Q->get];
      Q->get = (Q->get + 1) % Q->nbuckets;
      Q->nadded--;
      return v;
    }
    

    void    Reset(FIFOQueue *Q){
      Q->put = Q->get = 0;
      Q->nadded = 0;
    }


  } //end FIFOQueue namespace
} //end bia namespace

