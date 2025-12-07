#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stackandfifo.h"

Stack * StackNew(int n) {
  Stack *s;

  s = (Stack *) malloc(sizeof(Stack));
  if (!s) return 0;

  s->n   = n;
  s->top = -1;
  s->data = (int *) malloc(sizeof(int) * n);
  if (!s->data) {
    free(s);
    return 0;
  }
  return s;
}

void    StackDestroy(Stack *s) {
  if (s) {
    if (s->data) free(s->data);
    free(s);
  }
}

void    StackPush(Stack *s, int n) {
  s->data[++(s->top)] = n;
}

int     StackPop(Stack *s) {
  if (s->top == -1) return -1;
  return(s->data[s->top--]);
}

int     StackEmpty(Stack *s) {
  return(s->top == -1);
}

FIFOQ * FIFOQNew(int n) {
  FIFOQ *q;

  q = (FIFOQ *) malloc(sizeof(FIFOQ));
  if (!q) return 0;

  q->n   = n;
  q->put = q->get = 0;
  q->data = (int *) malloc(sizeof(int) * n);
  if (!q->data) {
    free(q);
    return 0;
  }
  return q;
}

void    FIFOQDestroy(FIFOQ *q) {
  if (q) {
    if (q->data) free(q->data);
    free(q);
  }  
}

void    FIFOQPush(FIFOQ *q, int n) {
  q->data[q->put] = n;
  q->put = (q->put + 1) % q->n;
}

int     FIFOQPop(FIFOQ *q) {
  int v;
  if (q->get == q->put) return -1;
  v = q->data[q->get];
  q->get = (q->get + 1) % q->n;
  return v;
}

int     FIFOQEmpty(FIFOQ *q) {
  return(q->get == q->put);
}

void    FIFOQReset(FIFOQ *q){
 q->put = q->get = 0;
}
