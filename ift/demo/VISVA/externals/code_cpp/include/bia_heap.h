
#ifndef _BIA_HEAP_H_
#define _BIA_HEAP_H_

#include "bia_common.h"

#define MINVALUE    0
#define MAXVALUE    1


namespace bia{
  namespace Heap{

    typedef struct _heap {
      float *cost;
      char *color;
      int *spel;
      int *pos;
      int last;
      int n;
      char removal_policy; //!< 0 is MINVALUE and 1 is MAXVALUE.
    } Heap;


    inline int GetDad(int i);
    inline int GetLeftSon(int i);
    inline int GetRightSon(int i);

    Heap *Create(int n, float *cost);
    void  Destroy(Heap **H);

    void  SetTheRemovalPolicy(Heap *H, char policy);
    bool  IsFull(Heap *H);
    bool  IsEmpty(Heap *H);

    bool  Insert(Heap *H, int spel);
    bool  Remove(Heap *H, int *spel);
    void  Update(Heap *H, int p, float value);
    void  GoUp(Heap *H, int i);
    void  GoDown(Heap *H, int i);
    void  Reset(Heap *H);


  } //end Heap namespace
} //end bia namespace

#endif



