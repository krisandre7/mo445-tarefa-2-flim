#ifndef msp_pq_h
#define msp_pq_h 1

namespace MSP{

//! Abstract priority queue interface
class PriorityQueue {
 public:
  /** Tag: element was never queued */
  static   int White = 0;
  /** Tag: element is queued */ 
  static   int Gray  = 1;
  /** Tag: element was queued and dequeued */
  static   int Black = 2;
  /** Internal list tag */ 
  static   int Nil   = -1;

  virtual ~PriorityQueue() { }

  //! Returns true if there are no elements queued
  virtual bool empty()=0;

  //! Empties the queue and turn all elements White
  virtual void reset()=0;

  //! Inserts element elem into the given bucket
  virtual void insert(int elem, int bucket)=0;

  //! Removes and returns the next element to be dequeued
  virtual int  remove()=0;

  //! Moves element elem from bucket <b>from</b> to bucket <b>to</b>
  virtual void update(int elem, int from, int to)=0;

  //! Returns the color tag of an element
  virtual int colorOf(int elem)=0;
};

//! Basic linear-time priority queue
class BasicPQ : public PriorityQueue {
 public:

  //! Constructor, creates a priority queue for elements 0..nelem-1 and
  //! up to nbucket priority buckets
  BasicPQ(int nelem, int nbucket) {
    NB = nbucket;
    NE = nelem;
    first = new int[NB];
    last  = new int[NB];
    prev = new int[NE];
    next = new int[NE];
    color = new char[NE];

    reset();
  }

  //! Destructor
  virtual ~BasicPQ() {
    if (first) delete first;
    if (last) delete last;
    if (prev) delete prev;
    if (next) delete next;
    if (color) delete color;
  }

  //! Returns true if the queue is empty
  bool empty() {
    int i;
    for(i=current;i<NB;i++)
      if (first[i] != Nil)
	return false;
    return true;
  }

  //! Empties the queue and turn all elements White
  void reset() { 
    int i;
    current = 0;

    for(i=0;i<NB;i++)
      first[i] = last[i] = Nil;

    for(i=0;i<NE;i++) {
      prev[i] = next[i] = Nil;
      color[i] = White;
    }
  }

  //! Inserts element elem into the given bucket
  void insert(int elem, int bucket) { 
    if (first[bucket] == Nil) {
      first[bucket] = elem;
      prev[elem] = Nil;
    } else {
      next[last[bucket]] = elem;
      prev[elem] = last[bucket];
    }
    last[bucket] = elem;
    next[elem] = Nil;
    color[elem] = Gray;
    if (bucket < current) current = bucket;
  }

  //! Removes and returns the next element to be dequeued
  int remove() { 
    int elem,n;
    while(first[current]==Nil && current < NB)
      ++current;
    if (current == NB) return Nil;
    elem = first[current];

    n = next[elem];
    if (n==Nil) {
      first[current] = last[current] = Nil;
    } else {
      first[current] = n;
      prev[n] = Nil;
    }
    color[elem] = Black;
    return elem;
  }

  //! Moves element elem from bucket <b>from</b> to bucket <b>to</b>
  void update(int elem, int from, int to) { 
    removeElem(elem,from);
    insert(elem,to);
  }

  //! Returns the color tag of an element
  int colorOf(int elem) { return((int)(color[elem])); }

 private:
  int NB,NE,current,*first, *last,*prev,*next;
  char *color;

  void removeElem(int elem, int bucket) {
    int p,n;
    
    p = prev[elem];
    n = next[elem];

    color[elem] = Black;
    
    if (first[bucket] == elem) {
      first[bucket] = n;
      if (n==Nil)
	last[bucket] = Nil;
      else
	prev[n] = Nil;
    } else {
      next[p] = n;
      if (n==Nil)
	last[bucket] = p;
      else
	prev[n] = p;
    }
  }

};

} //end MSP namespace

#endif
