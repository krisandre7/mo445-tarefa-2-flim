
#include "livewire.h"

namespace Interactive{

  int *ConfirmedPathArray(int *path, int src){
    int *cpath=NULL;
    int n,nn,i,j=NIL;
    if(path!=NULL){
      n = path[0];
      for(i=1; i<=n; i++){
	if(path[i]==src){
	  j = i;
	  break;
	}
      }
      if(j!=NIL){
	nn = n-j+1;
	if(nn>=2){
	  cpath = AllocIntArray(nn+1);
	  cpath[0] = nn;
	  for(i=1; j<=n; j++,i++){
	    cpath[i] = path[j];
	  }
	}
      }
    }    
    return(cpath);
  }


  int *Path2Array(Image *pred, int init, int dst){
    int *path=NULL,i,n,p;
  
    if(init==dst) dst = pred->val[dst];

    p = dst;
    n = 0;
    while(p!=NIL){
      n++;
      if(p==init) break;
      p = pred->val[p];
    }
    path = AllocIntArray(n+1);
    path[0]=n;
    p = dst;
    i = 0;
    while(p!=NIL){
      i++;
      path[i]=p;
      if(p==init) break;
      p = pred->val[p];
    }
    
    return(path);
  }


  int *path_by_iftLiveWire(Image *cost, Image *pred,
			   Image *arcw, AdjRel *A, int Wmax,
			   PriorityQueue **pQ,
			   int init, int src, int dst){
    PriorityQueue *Q = NULL;
    int i,p,q,n,weight;
    int tmp;
    Pixel u,v;
    int *path=NULL;
    
    if(pQ==NULL) return NULL;
    Q = *pQ;
    
    if(pred->val[dst]==NIL){
      
      n = arcw->ncols*arcw->nrows;

      if(Q==NULL){
	*pQ = CreatePQueue(ROUND(powf(Wmax,1.5))+1,n,cost->val);
	Q = *pQ;
      }

      if(IsEmptyPQueue(Q)){
	if(Q->L.elem[src].color != BLACK){
	  cost->val[src] = 0;
	  InsertElemPQueue(pQ, src);
	}
      }
      
      while(!IsEmptyPQueue(Q)){
	p = RemoveMinFIFOPQueue(Q);

	if (p==dst){
	  InsertElemPQueue(pQ, p);
	  break;
	}
	u.x = p%arcw->ncols;
	u.y = p/arcw->ncols;
	for(i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  if(ValidPixel(pred,v.x,v.y)){
	    q = v.x + pred->tbrow[v.y];
	    if(Q->L.elem[q].color != BLACK){

	      weight = (arcw->val[p]+arcw->val[q])/2;
	      weight = ROUND(powf(weight,1.5));
	      tmp = cost->val[p] + weight;
	      
	      if(tmp < cost->val[q]){
		if(Q->L.elem[q].color == GRAY)
		  RemoveElemPQueue(Q,q);
		cost->val[q] = tmp;
		pred->val[q] = p;
		InsertElemPQueue(pQ, q);
	      }
	    }
	  }
	}
      }
    }
    path = Path2Array(pred,init,dst);
    return(path);
  }



  void path_clear_current(Image *cost, Image *pred,
			  PriorityQueue **pQ,
			  int src){
    int n,p;
    n = cost->ncols*cost->nrows;
    
    if(*pQ!=NULL)
      DestroyPQueue(pQ);
    *pQ = NULL;
    for(p=0; p<n; p++){
      if(cost->val[p]>INT_MIN && p!=src){
	cost->val[p] = INT_MAX;
	pred->val[p] = NIL;
      }
    }
  }


} //end Interactive namespace


