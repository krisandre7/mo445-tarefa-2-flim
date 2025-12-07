#include "oldift.h"

typedef struct {
  int size;
  int capacity;
  int *l;
} orderlist;

orderlist *CreateOrderList( int capacity );
void OrderListAdd( orderlist *l, int k );
void OrderListRemove( orderlist *l, int k );
void OrderListReplace( orderlist *l, int in, int out );
int OrderListMedian( orderlist *l );
bool OrderListIsFull( orderlist *l );
void DestroyOrderList( orderlist **l );

Curve *MedianFilterHistogram( Curve *h );
int GetPeak( Curve *h, int xmin, int xmax );
int GetFirstPeak( Curve *h, int xmin, int xmax );
int GetLastPeak( Curve *h, int xmin, int xmax );
int GetValley( Curve *h, int xmin, int xmax );
int GetFirstValley( Curve *h, int xmin, int xmax );
int GetLastValley( Curve *h, int xmin, int xmax );
// Get first occurrence of a bin that is greater than Y[ s ] * ( 1 + T ) or lower than Y[ s ] * ( 1 - T ), from s to 0.
int GetPlateauBegining( Curve *h, int s, float T );
// Get first occurrence of a bin that is greater than Y[ s ] * ( 1 + T ) or lower than Y[ s ] * ( 1 - T ), from s to h->n - 1.
int GetPlateauEnding( Curve *h, int s, float T );
Curve *RemoveEmptyBins3( Curve *h );
