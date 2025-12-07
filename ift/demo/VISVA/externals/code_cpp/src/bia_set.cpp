
#include "bia_set.h"

namespace bia{
  namespace Set{

    Set *Create(){
      return NULL;
    }

    void Destroy(Set **S){
      Set *p;
      while(*S != NULL){
	p = *S;
	*S = p->next;
	free(p);
      }
    }

    //Set *Clone(Set *S){
    //  Set *tmp = NULL;
    //  Set *C = NULL;
    //  int p;
    //
    //  tmp = S;
    //  while(tmp!=NULL){
    //    p = tmp->elem;
    //    InsertSet(&C, p);
    //    tmp = tmp->next;
    //  }
    //  return C;
    //}

    // Copies the set in an ordered fashion
    Set *Clone(Set *S){
      Set *tmp = NULL;
      Set *C = NULL;
      Set **tail = NULL;
      int p;

      tmp = S;

      if(tmp != NULL){
  	p = tmp->elem;
	C = (Set *) calloc(1,sizeof(Set));
	C->elem = p;
	C->next = NULL;
	tail = &(C->next);
	tmp = tmp->next;
      }

      while(tmp!=NULL){
  	p = tmp->elem;
	*tail = (Set *) calloc(1,sizeof(Set));
	(*tail)->elem = p;
	(*tail)->next = NULL;
	tail = &((*tail)->next);
	tmp = tmp->next;
      }
      return C;
    }


    void Insert(Set **S, int elem){
      Set *p=NULL;
      
      p = (Set *) calloc(1,sizeof(Set));
      if(p == NULL) 
	bia::Error((char *)MSG1,
		   (char *)"Set::Insert");
      if(*S == NULL){
	p->elem  = elem;
	p->next  = NULL;
      }else{
	p->elem  = elem;
	p->next  = *S;
      }
      *S = p;
    }


    int Remove(Set **S){
      Set *p;
      int elem = NIL;

      if(*S != NULL){
	p    =  *S;
	elem = p->elem;
	*S   = p->next;
	//printf("RemoveSet before free");
	free(p);
	//printf(" RemoveSet after free: elem is %d\n",elem);
	//if(*S != NULL) printf(" *S->elem is %d\n",(*S)->elem);
      }
      return(elem);
    }


    void RemoveElem(Set **S, int elem){
      Set *tmp = NULL, *remove;
      
      tmp = *S;
      if ( tmp->elem == elem ) {
	*S = tmp->next;
	free( tmp );
      }
      else {
	while( tmp->next->elem != elem ) tmp = tmp->next;
	remove = tmp->next;
	tmp->next = remove->next;
	free( remove );
      }
    }


    bool IsInSet(Set *S, int elem){
      bool flag = false;
      
      while(S!=NULL){
	if(S->elem == elem){
	  flag = true;
	  break;
	}
	S = S->next;
      }
      return flag;
    }


    int     MinimumValue(Set *S){
      Set *aux;
      int p,min=INT_MAX;

      aux = S;
      while(aux != NULL){
	p = aux->elem;
	if(p<min) min = p;
	aux = aux->next;
      }
      return min;
    }


    int     MaximumValue(Set *S){
      Set *aux;
      int p,max=INT_MIN;
      
      aux = S;
      while(aux != NULL){
	p = aux->elem;
	if(p>max) max = p;
	aux = aux->next;
      }
      return max;
    }


    void    Convert2DisjointSets(Set **S1,
				 Set **S2){
      BMap::BMap *bmap;
      Set **S[2];
      Set *Cur,*Prev;
      int elem,i,max,min,p,n;
      
      if(*S1==NULL || *S2==NULL)
	return;
      
      min = MIN(MinimumValue(*S1),
		MinimumValue(*S2));
      
      max = MAX(MaximumValue(*S1),
		MaximumValue(*S2));

      n = max-min+1;
      bmap = BMap::Create(n);
      S[0] = S1;
      S[1] = S2;

      for(i=0; i<2; i++){
	Prev = Cur = *(S[i]);
	while(Cur != NULL){
	  elem = Cur->elem;
	  p = elem - min;
	  if( BMap::Get(bmap, p) ){
	    if(Prev==Cur){
	      *(S[i]) = Cur->next;
	      free(Cur);
	      Prev = Cur = *(S[i]);
	    }
	    else{
	      Prev->next = Cur->next;
	      free(Cur);
	      Cur = Prev->next;
	    }
	  }
	  else{
	    BMap::Set1(bmap, p);
	    Prev = Cur;
	    Cur = Cur->next;
	  }
	}
      }
      BMap::Destroy(&bmap);
    }


    int  GetNElems(Set *S){
      Set *aux;
      int size=0;

      aux = S;
      while(aux != NULL){
	size++;
	aux = aux->next;
      }
      return size;
    }


    void Merge(Set **S, Set **T) {
      Set *aux;
      if( *S == NULL ) *S = *T;
      else{
	for( aux = *S; aux->next != NULL; aux = aux->next );
	aux->next = *T;
      }
    }


  } //end Set namespace
} //end bia namespace

