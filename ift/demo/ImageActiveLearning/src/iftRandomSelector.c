#include "iftRandomSelector.h"

iftRandomSelector * iftCreateRandomSelector(  int size)
{
  iftRandomSelector * rs = (iftRandomSelector *) calloc(1, sizeof(iftRandomSelector));
  rs->nums = iftAllocIntArray(size);
  rs->currentSize = rs->totalSize = size;

  return rs;
}

iftRandomSelector * iftCreateRandomSelectorDefaultValues(  int size)
{
  iftRandomSelector * rs = iftCreateRandomSelector(size);
  for (int i = 0; i < size; ++i)
    rs->nums[i] = i;

  return rs;
}

iftRandomSelector ** iftCreateRandomSelectorArray(  int arraySize,   int *sizes)
{
  iftRandomSelector ** rsrs = (iftRandomSelector **) calloc(arraySize, sizeof(iftRandomSelector *));
  for (int i = 0; i < arraySize; ++i)
    rsrs[i] = iftCreateRandomSelector(sizes[i]);

  return rsrs;
}

int iftPickFromRandomSelector(iftRandomSelector * rs,   bool hasReposition)
{
  if (rs->currentSize <= 0)
    iftError("Random Selector pool is already empty", "iftPickFromRandomSelector");

  int i = iftRandomInteger(0, rs->currentSize - 1);
  int val = rs->nums[i];

  if(!hasReposition) {
    iftSwap(rs->nums[i], rs->nums[rs->currentSize - 1]);
    rs->currentSize -= 1;
  }

  return val;
}

void iftResetRandomSelector(iftRandomSelector * rs)
{
  rs->currentSize = rs->totalSize;
}

void iftDestroyRandomSelector(iftRandomSelector ** rs)
{
  if (rs != NULL) {
    iftRandomSelector *aux = *rs;
    if (aux != NULL) {
      if (aux->nums != NULL)
        free(aux->nums);
    }
    free(aux);
    *rs = NULL;
  }
}

void iftDestroyRandomSelectorArray(iftRandomSelector *** rsrs,   int size)
{
  if (rsrs != NULL) {
    iftRandomSelector **rs = *rsrs;
    for (int i = 0; i < size; ++i)
      if (rs[i] != NULL)
        iftDestroyRandomSelector(&(rs[i]));
    free(rs);
    *rsrs = NULL;
  }
}

