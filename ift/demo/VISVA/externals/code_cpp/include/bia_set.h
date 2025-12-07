
#ifndef _BIA_SET_H_
#define _BIA_SET_H_

#include "bia_common.h"
#include "bia_bmap.h"

namespace bia{
  namespace Set{

    typedef struct _set {
      int elem;
      struct _set *next;
    } Set;

    
    Set *Create();
    void Destroy(Set **S);
    Set *Clone(Set *S);

    void Insert(Set **S, int elem);
    int  Remove(Set **S);
    void RemoveElem(Set **S, int elem);
    bool IsInSet(Set *S, int elem);
    int  MinimumValue(Set *S);
    int  MaximumValue(Set *S);
    void Convert2DisjointSets(Set **S1,
			      Set **S2);
    int  GetNElems(Set *S);
    
    /**
     * \brief Merge two sets. 
     *
     * The next field of the last element of set S 
     * points to the first element of set T. 
     * T does not change.
     */
    void Merge(Set **S, Set **T);

  } //end Set namespace
} //end bia namespace

#endif

