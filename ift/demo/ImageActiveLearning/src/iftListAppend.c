#include "iftListAppend.h"

int iftRemoveListNode(iftList *L, iftNode **NodeRef, bool isReverse)
{
  if (L == NULL || NodeRef == NULL)
    iftError("List L or node NodeRef is NULL.", "iftExtractListNode");

  iftNode *N = *NodeRef;
  if (N == NULL)
    iftError("Trying to extract NULL node.", "iftExtractListNode");

  // Update client pointer
  if (isReverse)
    *NodeRef = N->previous;
  else
    *NodeRef = N->next;

  // Remove intended node from List
  if (N->previous != NULL)
    N->previous->next = N->next;
  else
    L->head = N->next;

  if (N->next != NULL)
    N->next->previous = N->previous;
  else
    L->tail = N->previous;

  // Extract value
  int elem = N->elem;
  L->n--;

  free(N);
  return elem;
}

iftListArray * iftCreateListArray(  int arraySize)
{
  iftListArray * la = calloc(1, sizeof(iftListArray));

  la->size = arraySize;
  la->list = (iftList **) calloc(arraySize, sizeof(iftList *));
  for (int i = 0; i < arraySize; ++i)
    la->list[i] = iftCreateList();

  return la;
}

void iftDestroyListArray(iftListArray ** listArray)
{
  if (listArray != NULL) {
    iftListArray * la = *listArray;
    if (la != NULL) {
      for (int i = 0; i < la->size; ++i) {
        iftDestroyList(&(la->list[i]));
      }
      free(la->list);
      free(la);
      *listArray = NULL;
    }
  }
}
