#include <ift.h>
#include "shared/iftListAppend.c"

void iftCompactPrintList(iftList *l)
{
  printf("List:");
  for (iftNode *n = l->head; n != NULL; n = n->next) {
    printf(" %d", n->elem);
  }
  printf("\n");
}

int main(int argc, char* argv[])
{
  size_t mem1 = iftMemoryUsed();

  int val;

  printf("-- Testing iftRemoveListNode\n");

  // Create list with values [0,9]
  iftList *list = iftCreateList(); 
  for(int i = 0; i < 10; ++i)
    iftInsertListIntoTail(list, i);
  iftCompactPrintList(list);

  // Remove head node
  val = iftRemoveListNode(list, &list->head, false);
  printf("Removing head: %d\n", val);
  iftCompactPrintList(list);

  // Remove tail node
  val = iftRemoveListNode(list, &list->tail, false);
  printf("Removing tail: %d\n", val);
  iftCompactPrintList(list);

  // Remove middle node
  val = iftRemoveListNode(list, &list->head->next, false);
  printf("Removing single from middle: %d\n", val);
  iftCompactPrintList(list);

  // Remove two middle nodes with reverse iterator
  printf("Removing from middle with reverse iterator:");
  iftNode * nodeIter = list->head->next->next;
  val = iftRemoveListNode(list, &nodeIter, true);
  printf(" %d", val);
  val = iftRemoveListNode(list, &nodeIter, true);
  printf(" %d\n", val);
  iftCompactPrintList(list);

  // Remove all elems with forward iterator
  nodeIter = list->head;
  printf("Removing all with forward iterator:");
  while(nodeIter != NULL) {
    val = iftRemoveListNode(list, &nodeIter, false);
    printf(" %d", val);
  }
  printf("\n");
  iftCompactPrintList(list);

  iftDestroyList(&list);

  printf("-- Testing ListArray\n");

  iftListArray *la = iftCreateListArray(3);
  val = 1;
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
      iftInsertListIntoTail(la->list[i],val++);
  for(int i = 0; i < 3; ++i) {
    printf("ListArray[%d]:", i);
    iftCompactPrintList(la->list[i]);
  }

  iftDestroyListArray(&la);

  size_t mem2 = iftMemoryUsed();

  iftVerifyMemory(mem1, mem2);

  return 0;
}
