#ifndef _IFT_RANDOM_SELECTOR_H_
#define _IFT_RANDOM_SELECTOR_H_

#include <ift.h>

typedef struct ift_random_selector {
  int * nums;
  int totalSize;
  int currentSize; // index for random selection without reposition
} iftRandomSelector;

iftRandomSelector * iftCreateRandomSelector(  int size);
iftRandomSelector * iftCreateRandomSelectorDefaultValues(  int size);
iftRandomSelector ** iftCreateRandomSelectorArray(  int arraySize,   int *sizes);
int iftPickFromRandomSelector(iftRandomSelector * rs,   bool hasReposition);
void iftResetRandomSelector(iftRandomSelector * rs);
void iftDestroyRandomSelector(iftRandomSelector ** rs);
void iftDestroyRandomSelectorArray(iftRandomSelector *** rsrs,   int size);

#endif // _IFT_RANDOM_SELECTOR_H_
