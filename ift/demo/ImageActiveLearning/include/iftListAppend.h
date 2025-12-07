/**
 * @file
 * @details TODO Finish documentation and move to iftList.[c/h].
 */
#ifndef _IFT_LISTAPPEND_H_
#define _IFT_LISTAPPEND_H_

#include <ift.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _ift_list_array {
  int size;
  iftList ** list;
} iftListArray;

/**
 * @brief Remove node from list an return its value.
 * @author Felipe Lemes
 * @date November 12, 2016
 *
 * Given that node pointed by <NodeRef> is part of list <L>.
 * Extracts integer from that node and removes it from the list.
 * If <NodeRef> is an external iterator (i.e. not a pointer managed by the list), 
 *   moves <NodeRef> to the next or previous node according to <isReverse>.
 * If the iftList or the iftNode are not allocated, an error will be raised.
 * If the node is not part of <L>, behavior is undefined.
 *
 * @param L The iftList.
 * @param NodeRef Reference to a node from L. Moved as side-effect.
 * @param isReverse Flag indicating if N will be moved backwards through list.
 *
 * @return Integer from removed node.
 */
int iftRemoveListNode(iftList *L, iftNode **NodeRef, bool isReverse);

iftListArray * iftCreateListArray(  int arraySize);

void iftDestroyListArray(iftListArray ** listArray);

#ifdef __cplusplus
}
#endif

#endif // _IFT_LISTAPPEND_H_
