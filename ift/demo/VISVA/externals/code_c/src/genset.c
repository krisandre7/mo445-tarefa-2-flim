#include "genset.h"

void InsertGenSet(GenSet **S, void *elem){
  GenSet *p=NULL;

  p = (GenSet *) calloc(1,sizeof(GenSet));
  if (p == NULL) Error(MSG1,"InsertGenSet");
  if (*S == NULL){
    p->elem  = elem;
    p->next  = NULL;
  }else{
    p->elem  = elem;
    p->next  = *S;
  }
  *S = p;
}

void *RemoveGenSet(GenSet **S){
  GenSet *p;
  void *elem=NULL;
  
  if (*S != NULL){
    p    =  *S;
    elem = p->elem;
    *S   = p->next;
    free(p);
  }

  return(elem);
}

void DestroyGenSet(GenSet **S){
  GenSet *p;
  while(*S != NULL){
    p = *S;
    *S = p->next;
    free(p->elem);
    free(p);
  }
}

bool IsInGenSet(GenSet *S, void *elem){
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



