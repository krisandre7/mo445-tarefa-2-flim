#include "ift.h"
#include "shared/iftRandomSelector.c"

#define TEST_SIZE 20

int main(int argc, char* argv[])
{
  size_t mem1 = iftMemoryUsed();

  // Single random selector test
  iftRandomSeed(time(NULL));
  iftRandomSelector *rs = iftCreateRandomSelector(TEST_SIZE);
  printf("Testing Random Selector with values from 0 to %d\n", TEST_SIZE);
  for(int i = 0; i < rs->currentSize; ++i)
    rs->nums[i] = i;
  printf("Printing %d values with reposition:\n", TEST_SIZE);
  for(int i = 0; i < TEST_SIZE; ++i)
    printf(" %d", iftPickFromRandomSelector(rs, TRUE));
  printf("\nPrinting %d values without reposition:\n", TEST_SIZE);
  for(int i = 0; i < TEST_SIZE; ++i)
    printf(" %d", iftPickFromRandomSelector(rs, FALSE));
  printf("\n");

  iftDestroyRandomSelector(&rs);

  // Random selector array test
  int sizes[5] = {5,5,5,5,5};
  iftRandomSelector **rsrs = iftCreateRandomSelectorArray(5, sizes);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      rsrs[j]->nums[i] = i;
  printf("Testing 5x5 Random Selector Array:\n");
  printf("Each column corresponds to a random selector with values [0,4] without reposition\n");
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j)
      printf(" %d", iftPickFromRandomSelector(rsrs[j], FALSE));
    printf("\n");
  }

  iftDestroyRandomSelectorArray(&rsrs, 5);

  size_t mem2 = iftMemoryUsed();
  iftVerifyMemory(mem1, mem2);

  return 0;
}
