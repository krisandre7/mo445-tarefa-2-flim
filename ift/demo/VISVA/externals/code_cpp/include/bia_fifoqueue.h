
#ifndef _BIA_FIFOQUEUE_H_
#define _BIA_FIFOQUEUE_H_

#include "bia_common.h"

namespace bia{
  namespace FIFOQueue{

    /**
     * \brief FIFO Queue with circular and growing features.
     */
    typedef struct _fifoqueue {
      int *data;
      int get, put;
      int nbuckets;
      int nadded;   //!< Number of elements added.
    } FIFOQueue;


    FIFOQueue  *Create(int nbuckets);
    void        Destroy(FIFOQueue **Q);
    
    void        Push(FIFOQueue *Q, int p);

    /**
     * @return Returns NIL if empty.
     */
    int         Pop(FIFOQueue *Q);
    void        Reset(FIFOQueue *Q);
    inline bool IsEmpty(FIFOQueue *Q){ 
      return (Q->nadded==0); 
    }
    inline bool IsFull(FIFOQueue *Q){ 
      return (Q->nadded==Q->nbuckets); 
    }
    
  } //end FIFOQueue namespace
} //end bia namespace

#endif


