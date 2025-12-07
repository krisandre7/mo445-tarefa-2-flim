
#ifndef _BIA_STACK_H
#define _BIA_STACK_H 1

#include "bia_common.h"

namespace bia{
  namespace Stack{

    typedef struct _stack {
      int *data;
      int top;
      int n;
    } Stack;
    
    Stack *Create(int n);
    void   Destroy(Stack **S);
    void   Push(Stack *S, int p);
    /**
     * @return Returns NIL if empty.
     */
    int    Pop(Stack *S);

    inline int IsEmpty(Stack *S){ return(S->top == -1); }

  } //end Stack namespace
} //end bia namespace

#endif
