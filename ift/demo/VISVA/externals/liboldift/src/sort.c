#include "sort.h"

void SelectionSort(int *val, int nelems, char order)
{
  int i,j,jmax;

  if (order == INCREASING){
    for (i=nelems-1; i > 0; i--){ 
      jmax = i;
      for (j=0; j < i; j++){
	if (val[j] > val[jmax])
	  jmax = j;
      }
      // if (jmax != i) // Not needed. Normally faster without the if.
      Change(&val[i],&val[jmax]);
    }
  } else { /* DECREASING */
    for (i=0; i < nelems-1; i++){ 
      jmax = i;
      for (j=i+1; j < nelems; j++){
	if (val[j] > val[jmax])
	  jmax = j;
      }
      //if (jmax != i) // Not needed. Normally faster without the if.
      Change(&val[i],&val[jmax]);
    }
  }
}

int *BucketSort(int *val, int nelems, char order)
{
  int i,j,maxval=INT_MIN,*sval=NULL;
  Queue *Q=NULL;

  for(i=0; i < nelems; i++) 
    if (val[i] > maxval)
      maxval = val[i];
  
  Q = CreateQueue(maxval+1,nelems);
  for(i=0; i < nelems; i++)
    InsertQueue(Q,val[i],i);

  sval = AllocIntArray(nelems);
  if (order==INCREASING){
    j = 0;
    while(!EmptyQueue(Q)) {
      i = RemoveQueue(Q);
      sval[j] = val[i];
      j++;
    }
  } else { /* order = DECREASING */
    j = nelems-1;
    while(!EmptyQueue(Q)) {
      i = RemoveQueue(Q);
      sval[j] = val[i];
      j--;
    }  
  }

  DestroyQueue(&Q);
  
  return(sval);
}

/* Bug at the pivot indexing correct - Giovani - 2011/04/13 */
void IndexQuickSort( int *value, int *index, int i0, int i1, char order ) {
  int m, d;
  
  if( i0 < i1 ) {
    /* random index to avoid bad pivots.*/
    d = RandomInteger( i0, i1 );
    Change( &value[ d ], &value[ i0 ] );
    Change( &index[ d ], &index[ i0 ] );
    m = i0;

    if( order == INCREASING ) {
      for( d = i0 + 1; d <= i1; d++ ) {
	if( value[ d ] <= value[ i0 ] ) {
	  m++;
	  Change( &value[ d ], &value[ m ] );
	  Change( &index[ d ], &index[ m ] );
	}
      }
    }
    else {
      for( d = i0 + 1; d <= i1; d++ ) {
	if( value[ d ] >= value[ i0 ] ) {
	  m++;
	  Change( &value[ d ], &value[ m ] );
	  Change( &index[ d ], &index[ m ] );
	}
      }
    }
    Change( &value[ m ], &value[ i0 ] );
    Change( &index[ m ], &index[ i0 ] );
    IndexQuickSort( value, index, i0, m - 1, order );
    IndexQuickSort( value, index, m + 1, i1, order );
  }
}

/* Bug at the pivot indexing correct - Giovani - 2011/04/13 */
void FloatQuickSort( float *value, int *index, int i0, int i1, char order ) {
  int m, d;
  
  if( i0 < i1 ) {
    /* random index to avoid bad pivots.*/
    d = RandomInteger( i0, i1 );
    FChange( &value[ d ], &value[ i0 ] );
    Change( &index[ d ], &index[ i0 ] );
    m = i0;

    if( order == INCREASING ) {
      for( d = i0 + 1; d <= i1; d++ ) {
	if( value[ d ] <= value[ i0 ] ) {
	  m++;
	  FChange( &value[ d ], &value[ m ] );
	  Change( &index[ d ], &index[ m ] );
	}
      }
    }
    else {
      for( d = i0 + 1; d <= i1; d++ ) {
	if( value[ d ] >= value[ i0 ] ) {
	  m++;
	  FChange( &value[ d ], &value[ m ] );
	  Change( &index[ d ], &index[ m ] );
	}
      }
    }
    FChange( &value[ m ], &value[ i0 ] );
    Change( &index[ m ], &index[ i0 ] );
    FloatQuickSort( value, index, i0, m - 1, order );
    FloatQuickSort( value, index, m + 1, i1, order );
  }
}

