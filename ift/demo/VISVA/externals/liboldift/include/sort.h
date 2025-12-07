#ifndef _SORT_H_
#define _SORT_H_

#include "queue.h"
#include "realheap.h"

int   *BucketSort(int *val, int nelems, char order);
void   SelectionSort(int *val, int nelems, char order); /* in place */
void IndexQuickSort( int *value, int *index, int i0, int i1, char order );
void FloatQuickSort( float *value, int *index, int i0, int i1, char order );
#endif
