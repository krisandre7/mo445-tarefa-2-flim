
#ifndef STACKANDFIFO_H
#define STACKANDFIFO_H 1

typedef struct ift_stack {
  int *data;
  int top;
  int n;
} Stack;

typedef struct ift_fifoq {
  int *data;
  int get, put;
  int n;
} FIFOQ;

Stack * StackNew(int n);
void    StackDestroy(Stack *s);
void    StackPush(Stack *s, int n);
int     StackPop(Stack *s); /* returns -1 if empty */
int     StackEmpty(Stack *s);

FIFOQ * FIFOQNew(int n);
void    FIFOQDestroy(FIFOQ *q);
void    FIFOQPush(FIFOQ *q, int n);
int     FIFOQPop(FIFOQ *q); /* returns -1 if empty */
int     FIFOQEmpty(FIFOQ *q);
void    FIFOQReset(FIFOQ *q);
#endif
