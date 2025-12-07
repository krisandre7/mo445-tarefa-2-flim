
#include "bia_stack.h"

namespace bia{
  namespace Stack{

    Stack *Create(int n) {
      Stack *S;
      S = (Stack *) malloc(sizeof(Stack));
      if(S==NULL) bia::Error((char *)MSG1,
			     (char *)"Stack::Create");
      S->n   = n;
      S->top = -1;
      S->data = bia::AllocIntArray(n);
      return S;
    }

    void    Destroy(Stack **S) {
      Stack *aux = *S;
      if(aux) {
	if(aux->data) bia::FreeIntArray(&aux->data);
	free(aux);
	*S = NULL;
      }
    }

    void    Push(Stack *S, int p) {
      S->data[++(S->top)] = p;
    }

    int     Pop(Stack *S) {
      if(S->top == -1) return -1;
      return(S->data[S->top--]);
    }

    
  } //end Stack namespace
} //end bia namespace

