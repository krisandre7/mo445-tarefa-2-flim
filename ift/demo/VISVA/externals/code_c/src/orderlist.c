#include "oldift.h"
#include "orderlist.h"

orderlist *CreateOrderList( int capacity ){
  orderlist *l;

  l = ( orderlist* ) calloc( 1, sizeof( orderlist ) );
  l->l = AllocIntArray( capacity );
  l->capacity = capacity;
  l->size = 0;

  return( l );
}

void OrderListAdd( orderlist *l, int k ) {
  int p;
  int tmp;

  for( p = 0; p < l->size; p++ ) {
    if( l->l[ p ] > k ) {
      tmp = k;
      k = l->l[ p ];
      l->l[ p ] = tmp;
    }
  }
  if( l->size < l->capacity ) {
    l->l[ p ] = k;
    l->size = l->size + 1;
  }
}

void OrderListRemove( orderlist *l, int k ) {
  int p;
  bool found;

  for( p = 0, found = false; p < l->size - 1; p++ ) {
    if( l->l[ p ] == k ) {
      found = true;
    }
    if( found == true )
      l->l[ p ] = l->l[ p + 1 ];
  }
  if( l->size != 0 )
    l->size = l->size - 1;
}

void OrderListReplace( orderlist *l, int in, int out ) {
  OrderListRemove( l, out );
  OrderListAdd( l, in );
}

int OrderListMedian( orderlist *l ) {
  if( l->size % 2 == 0 )
    return( ( l->l[ l->size / 2 ] + l->l[ l->size / 2 - 1 ] ) / 2 );
  else
    return( l->l[ l->size / 2 ] );
}
  

bool OrderListIsFull( orderlist *l ) {
  return( l->size == l->capacity );
}

void DestroyOrderList( orderlist **l ) {
  orderlist *aux;

  aux = *l;
  if( aux != NULL ) {
    if( aux->l != NULL )
      free( aux->l );
    free( aux );
  }
  *l = NULL;
}

Curve *MedianFilterHistogram( Curve *h ) {
  Curve *m = NULL;
  Curve *aux = NULL;
  orderlist *adj;
  int p;
  int kernel;
  
  kernel = 5 + ROUND( h->n / 1000 );

  adj = CreateOrderList( kernel );
  aux = RemoveEmptyBins3( h );
  m = CopyCurve( aux );
  aux->Y[ 0 ] = 0; // remove background bin in histogram.
  m->Y[ 0 ] = 0; // Setting 0 to background bin in median histogram.

  for( p = 0; p <= kernel / 2; p++ )
    OrderListAdd( adj, aux->Y[ p ] );
  for( p = kernel / 2 + 1; p < kernel; p++ ) {
    OrderListAdd( adj, aux->Y[ p ] );
    m->Y[ p - kernel / 2 ] = OrderListMedian( adj );
  }
  for( p = kernel; p < aux->n; p++ ) {
    OrderListReplace( adj, aux->Y[ p ], aux->Y[ p - kernel ] );
    m->Y[ p - kernel / 2 ] = OrderListMedian( adj );
  }
  for( p = aux->n; p < aux->n + kernel / 2; p++ ) {
    OrderListRemove( adj, aux->Y[ p - kernel ] );
    m->Y[ p - kernel / 2 ] = OrderListMedian( adj );
  }

  DestroyOrderList( &adj );
  DestroyCurve( &aux );
  return( m );
}

int GetPeak( Curve *h, int xmin, int xmax ) {
  int p, peak;
  peak = xmin;
  for( p = xmin + 1; p < xmax; p++ ) {
    if( h->Y[ peak ] < h->Y[ p ] ) {
      peak = p;
    }
  }
  return( peak );
}

int GetFirstPeak( Curve *h, int xmin, int xmax ) {
  int p, peak;
  int size, step;
  bool found;
  size = xmax - xmin;
  step = size / 10;
  if( step < 4 )
    step = 4;

  peak = xmin;
  for( p = xmin, found = false; ( found == false && p < xmax ); p++ ) {
    if( h->Y[ p ] > h->Y[ peak ] ) {
      peak = p;
    }
    else if( peak < p - step )
      found = true;
  }
  return( peak );
}

int GetLastPeak( Curve *h, int xmin, int xmax ) {
  int p, peak;
  int size, step;
  bool found;
  size = xmax - xmin;
  step = size / 10;
  if( step < 4 )
    step = 4;

  peak = xmax;
  for( p = xmax, found = false; ( found == false  && p > xmin ); p-- ) {
    if( h->Y[ p ] > h->Y[ peak ] )
      peak = p;
    else if( peak > p + step )
      found = true;
  }
  return( peak );
}

int GetValley( Curve *h, int xmin, int xmax ) {
  int p, valley;
  valley = xmin;
  for( p = xmin + 1; p < xmax; p++ ) {
    if( h->Y[ valley ] > h->Y[ p ] ) {
      valley = p;
    }
  }
  return( valley );
}

int GetFirstValley( Curve *h, int xmin, int xmax ) {
  int p, valley;
  int size, step;
  bool found;
  size = xmax - xmin;
  step = size / 10;
  if( step < 4 )
    step = 4;

  valley = xmin;
  for( p = xmin, found = false; ( found == false && p < xmax ); p++ ) {
    if( h->Y[ p ] < h->Y[ valley ] ) {
      valley = p;
    }
    else if( valley < p - step )
      found = true;
  }
  return( valley );
}

int GetLastValley( Curve *h, int xmin, int xmax ) {
  int p, valley;
  int size, step;
  bool found;
  size = xmax - xmin;
  step = size / 10;
  if( step < 4 )
    step = 4;

  valley = xmax;
  for( p = xmax, found = false; ( found == false  && p > xmin ); p-- ) {
    if( h->Y[ p ] < h->Y[ valley ] )
      valley = p;
    else if( valley > p + step )
      found = true;
  }
  return( valley );
  
}

// Get first occurrence of a bin that is greater than Y[ s ] * ( 1 + T ) or lower than Y[ s ] * ( 1 - T ), from s to 0.
int GetPlateauBegining( Curve *h, int s, float T ) {
  int p;
  
  for( p = s; ( h->Y[ s ] * ( 1 - T ) < h->Y[ p ] ) && ( h->Y[ s ] * ( 1 + T ) > h->Y[ p ] ); p-- ) {
    if( p == 0 )
      break;
  }
  return( p );
}

// Get first occurrence of a bin that is greater than Y[ s ] * ( 1 + T ) or lower than Y[ s ] * ( 1 - T ), from s to h->n - 1.
int GetPlateauEnding( Curve *h, int s, float T ) {
  int p;
  
  for( p = s; ( h->Y[ s ] * ( 1 - T ) < h->Y[ p ] ) && ( h->Y[ s ] * ( 1 + T ) > h->Y[ p ] ); p++ ) {
    if( p == h->n - 1 )
      break;
  }
  return( p );
}

Curve *RemoveEmptyBins3( Curve *h ) {
  Curve *r = NULL;
  int i, j, nonzeros;

  nonzeros = 0;
  for( i = 0; i < h->n; i++) {
    if( h->Y[ i ] != 0 )
      nonzeros++;
  }
  r = CreateCurve( nonzeros );
  
  for( i = 0, j = 0; i < h->n; i++) {
    if( h->Y[ i ] > 0 ) {
      r->X[ j ] = i;
      r->Y[ j ] = h->Y[ i ];
      j++;
    }
  }

  return( r );
}
